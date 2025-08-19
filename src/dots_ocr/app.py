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
import torch

load_dotenv()

DEVICE, DTYPE = pick_device_and_dtype()
MODEL_PATH = os.environ.get("MODEL_PATH", "./weights/DotsOCR")
ATTN_IMPL = os.environ.get("ATTN_IMPL", "sdpa")  # avoid flash-attn issues
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "24000"))


model = None
processor = None
generate_lock = asyncio.Lock()


app = FastAPI(title="DotsOCR FastAPI Server", version="1.0.0")

parse_document_prompt =  """Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
"""


class InferResponse(BaseModel):
    text: str
    latency_sec: float
    device: str

@app.on_event("startup")
def _startup():
    global model, processor
    use_multi_gpu = torch.cuda.device_count() > 1
    model, processor = load_pipeline(
        model_path=MODEL_PATH,
        attention_impl=ATTN_IMPL,
        device=DEVICE,
        dtype=DTYPE,
        use_multi_gpu=use_multi_gpu
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

async def _run_sync_inference(prompt: str, max_new_tokens: int, file: UploadFile) -> Dict[str, Any]:
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

@app.post("/v1/infer", response_model=InferResponse)
async def infer(
    prompt: str = Body(..., embed=True, description="Instruction/prompt for the model"),
    max_new_tokens: int = Body(DEFAULT_MAX_NEW_TOKENS, embed=True, ge=1, le=64000),
    file: UploadFile = File(...)
):
    return await _run_sync_inference(prompt, max_new_tokens, file)
    

@app.post("/v1/infer_parse", response_model = InferResponse)
async def infer_parse(
    max_new_tokens: int = Body(DEFAULT_MAX_NEW_TOKENS, embed=True, ge=1, le=64000),
    file: UploadFile = File(...)
):
    prompt = parse_document_prompt
    return await _run_sync_inference(prompt, max_new_tokens, file)

def main():
    import os, uvicorn
    uvicorn.run("dots_ocr.app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))

if __name__ == "__main__":
    # For local dev only; in prod use: uvicorn app:app --host 0.0.0.0 --port 8000
    main()