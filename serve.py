#!/usr/bin/env python3
"""
OpenAI-compatible inference server for MiniCPM-V.

Exposes POST /v1/chat/completions accepting text + base64 images,
returning JSON in OpenAI format. Used by the ERP extraction service
when USE_MINICPM_V=true.

Usage:
    python serve.py [--model openbmb/MiniCPM-V-2_6] [--port 8001] [--device cuda]

Environment variables (override CLI flags):
    MINICPM_MODEL   model path / HuggingFace repo id (default: openbmb/MiniCPM-V-2_6)
    MINICPM_PORT    port to listen on (default: 8001)
    MINICPM_DEVICE  cuda | cpu | mps (default: auto-detect)
"""

import argparse
import base64
import io
import json
import logging
import os
import time
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from transformers import AutoModel, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("minicpm-serve")

# ─── CLI / env config ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="MiniCPM-V OpenAI-compatible server")
parser.add_argument("--model",  default=os.environ.get("MINICPM_MODEL",  "openbmb/MiniCPM-V-2_6"))
parser.add_argument("--port",   default=int(os.environ.get("MINICPM_PORT",  "8001")), type=int)
parser.add_argument("--device", default=os.environ.get("MINICPM_DEVICE", "auto"))
parser.add_argument("--max-new-tokens", default=2048, type=int)
args = parser.parse_args()

# ─── Device selection ─────────────────────────────────────────────────────────

def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

DEVICE = resolve_device(args.device)
log.info(f"Using device: {DEVICE}")

# ─── Model loading ────────────────────────────────────────────────────────────

log.info(f"Loading model: {args.model}")

dtype = torch.bfloat16 if DEVICE in ("cuda", "mps") else torch.float32

model = AutoModel.from_pretrained(
    args.model,
    trust_remote_code=True,
    torch_dtype=dtype,
)
model = model.to(device=DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
log.info("Model loaded.")

# ─── Pydantic schemas (OpenAI subset) ─────────────────────────────────────────

class ImageUrl(BaseModel):
    url: str  # "data:image/jpeg;base64,<b64>" or http url

class ContentPart(BaseModel):
    type: str                        # "text" | "image_url"
    text: str | None = None
    image_url: ImageUrl | None = None

class Message(BaseModel):
    role: str
    content: str | list[ContentPart]

class ChatRequest(BaseModel):
    model: str = Field(default="minicpm-v")
    messages: list[Message]
    max_tokens: int = Field(default=2048)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.9)
    stream: bool = Field(default=False)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def decode_image(part: ContentPart) -> Image.Image | None:
    """Decode an image from a content part (base64 data URI or http URL)."""
    if part.type != "image_url" or part.image_url is None:
        return None
    url = part.image_url.url
    if url.startswith("data:"):
        # data:image/jpeg;base64,<data>
        header, data = url.split(",", 1)
        img_bytes = base64.b64decode(data)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Plain HTTP URL (requests needed)
    try:
        import requests as req
        resp = req.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        log.warning(f"Failed to fetch image from URL: {exc}")
        return None


def build_msgs_and_images(request: ChatRequest) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Convert OpenAI-style messages to MiniCPM-V format."""
    msgs: list[dict[str, Any]] = []
    images: list[Image.Image] = []

    for message in request.messages:
        if isinstance(message.content, str):
            msgs.append({"role": message.role, "content": message.content})
            continue

        text_parts: list[str] = []
        for part in message.content:
            if part.type == "text" and part.text:
                text_parts.append(part.text)
            elif part.type == "image_url":
                img = decode_image(part)
                if img is not None:
                    images.append(img)
                    text_parts.append("<image>")

        msgs.append({"role": message.role, "content": "\n".join(text_parts)})

    return msgs, images


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="MiniCPM-V inference server", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok", "model": args.model, "device": DEVICE}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    if request.stream:
        raise HTTPException(status_code=400, detail="Streaming not supported in this server.")

    msgs, images = build_msgs_and_images(request)

    try:
        if images:
            # Pass image list to the model; MiniCPM-V supports list of PIL images
            result = model.chat(
                image=images[0] if len(images) == 1 else images,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=max(request.temperature, 0.01),
                top_p=request.top_p,
                max_new_tokens=min(request.max_tokens, args.max_new_tokens),
            )
        else:
            result = model.chat(
                image=None,
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=max(request.temperature, 0.01),
                top_p=request.top_p,
                max_new_tokens=min(request.max_tokens, args.max_new_tokens),
            )
    except Exception as exc:
        log.exception("Inference error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # result is a string from model.chat()
    reply_text = result if isinstance(result, str) else str(result)

    return JSONResponse(content={
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": args.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": -1,
            "completion_tokens": -1,
            "total_tokens": -1,
        },
    })


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
