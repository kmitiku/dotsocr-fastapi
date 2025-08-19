import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from DotsOCR import configuration_dots


def load_pipeline(model_path: str, attention_impl: str, device: str, dtype: torch.dtype):
    global model, processor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation=attention_impl,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model, processor


