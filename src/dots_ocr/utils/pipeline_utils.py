import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from DotsOCR import configuration_dots


def load_pipeline(model_path: str, attention_impl: str, device: str, dtype: torch.dtype, use_multi_gpu: bool):
    global model, processor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if use_multi_gpu:
        max_memory = {}
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory  # bytes
            giB = int(total / (1024 ** 3) * 0.90)
            max_memory[i] = f"{giB}GiB"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation=attention_impl,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto",
            max_memory=max_memory,
        )
    else:

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


