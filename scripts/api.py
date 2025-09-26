"""FastAPI microservice exposing Temux text generation."""

from __future__ import annotations

import logging
import threading
import time
from functools import lru_cache
from typing import Dict, Iterable, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from src.temux_lite_50m import ensure_model_on_device

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "TheTemuxFamily/Temux-Lite-50M"


class GenerateRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    device: Optional[str] = None
    max_new_tokens: int = 196
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = True
@lru_cache(maxsize=None)
def _load_model(model_id: str, device: Optional[str] = None):
    LOGGER.info("Loading model %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    resolved_device = ensure_model_on_device(model, device)
    model.eval()
    LOGGER.info("Loaded %s on %s", model_id, resolved_device)
    return tokenizer, model, resolved_device


def _stream_generate(model, tokenizer, prompt: str, **generate_kwargs) -> Iterable[str]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(inputs, streamer=streamer, **generate_kwargs)
    thread = threading.Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    try:
        for token in streamer:
            yield token
    finally:
        thread.join()


def create_app(default_model_id: str = DEFAULT_MODEL_ID, preload: bool = False) -> FastAPI:
    app = FastAPI(title="Temux Inference API", version="0.1.0")
    app.state.default_model_id = default_model_id

    if preload:
        try:
            _load_model(default_model_id)
        except Exception as exc:  # pragma: no cover - logging path
            LOGGER.warning("Failed to preload model %s: %s", default_model_id, exc)

    @app.post("/generate")
    async def generate(request: GenerateRequest):  # type: ignore[override]
        model_id = request.model or app.state.default_model_id
        try:
            tokenizer, model, _ = _load_model(model_id, request.device)
        except Exception as exc:  # pragma: no cover - error path
            LOGGER.error("Unable to load model %s: %s", model_id, exc)
            raise HTTPException(status_code=500, detail=str(exc))

        generate_kwargs: Dict[str, object] = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": True,
        }

        if request.stream:
            def token_iterator() -> Iterable[bytes]:
                for token in _stream_generate(model, tokenizer, request.prompt, **generate_kwargs):
                    yield token.encode("utf-8")

            return StreamingResponse(token_iterator(), media_type="text/plain")

        inputs = tokenizer(request.prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if text.startswith(request.prompt):
            text = text[len(request.prompt):]
        return {"output": text.strip(), "latency_ms": latency_ms}

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import uvicorn

    uvicorn.run("scripts.api:app", host="0.0.0.0", port=8000, reload=False)
