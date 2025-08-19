

from qwen_vl_utils import process_vision_info
import torch
from typing import Any, Dict
import time
from fastapi import HTTPException
from transformers import ProcessorMixin, PreTrainedModel


def _prepare_messages(image_path: str, prompt: str) -> Dict[str, Any]:
    """
    Build the multi-modal chat messages payload that Qwen-VL/DotsOCR expects.
    image_path: str - The path to the image file (local or remote).
    prompt: str - The text prompt to accompany the image.
    """

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path
                },
                {"type": "text", "text": prompt}
            ]
        }
    ]


def __get_model_inputs(processor: ProcessorMixin, image_path: str, prompt: str , device: str | torch.device | None = None) -> Dict[str, Any]:
    messages = _prepare_messages(image_path, prompt)
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    return inputs


def get_generation_output(
        model: PreTrainedModel,
        processor: ProcessorMixin,
        image_path: str, prompt: str,
        max_new_tokens: int,
        device: str | torch.device | None = None
    ) -> Dict[str, Any]:

    start = time.time()
    if type(device) is str:
        device = torch.device(device)
    elif device is None:
        device = torch.device("cpu")
    elif not isinstance(device, torch.device):
        raise HTTPException(status_code=400, detail="Invalid device type.")

    if not hasattr(processor, "apply_chat_template"):
        raise HTTPException(
            status_code=500,
            detail="This processor has no chat template. Ensure your model/processor provides `apply_chat_template`."
        )
    



    inputs = __get_model_inputs(processor, image_path, prompt, device=device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    elapsed = time.time() - start

    # Free cache on GPU/MPS a bit
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return {
        "text": output_texts[0] if output_texts else "",
        "latency_sec": round(elapsed, 3),
        "device": device.type,
    }
