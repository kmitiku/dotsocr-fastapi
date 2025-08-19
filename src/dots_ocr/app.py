import os
import asyncio
from typing import Any, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from pydantic import BaseModel
from dots_ocr.utils.device_utils import pick_device_and_dtype
from dots_ocr.utils.pipeline_utils import load_pipeline
from dots_ocr.utils.inference_utils import get_generation_output
from dots_ocr.utils.io_utils import save_upload_to_temp as save_upload_to_temp
from dotenv import load_dotenv

load_dotenv()

DEVICE, DTYPE = pick_device_and_dtype()
MODEL_PATH = os.environ.get("MODEL_PATH", "./weights/DotsOCR")
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")  # avoid flash-attn issues
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "24000"))


model = None
processor = None
generate_lock = asyncio.Lock()


app = FastAPI(title="DotsOCR FastAPI Server", version="1.0.0")


class InferResponse(BaseModel):
    text: str
    latency_sec: float
    device: str

@app.on_event("startup")
def _startup():
    global model, processor
    model, processor = load_pipeline(
        model_path=MODEL_PATH,
        attention_impl=ATTN_IMPL,
        device=DEVICE,
        dtype=DTYPE
    )

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE.type,
        "dtype": str(DTYPE),
        "model_path": MODEL_PATH,
        "attn_implementation": ATTN_IMPL
    }

def _run_generation(image_path: str, prompt: str, max_new_tokens: int) -> Dict[str, Any]:
    global model, processor
    if not model or not processor:
        raise HTTPException(status_code=500, detail="Model and processor not loaded")

    return get_generation_output(
        model=model,
        processor=processor,
        image_path=image_path,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        device=DEVICE
    )

@app.post("/v1/infer-upload", response_model=InferResponse)
async def infer_upload(
    prompt: str = Body(..., embed=True, description="Instruction/prompt for the model"),
    max_new_tokens: int = Body(DEFAULT_MAX_NEW_TOKENS, embed=True, ge=1, le=64000),
    file: UploadFile = File(...)
):
    tmp_path = None
    try:
        tmp_path = save_upload_to_temp(file)
        async with generate_lock:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                _run_generation,
                tmp_path,
                prompt,
                max_new_tokens
            )
        return result
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
def main():
    import os, uvicorn
    uvicorn.run("dots_ocr.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

if __name__ == "__main__":
    # For local dev only; in prod use: uvicorn app:app --host 0.0.0.0 --port 8000
    main()