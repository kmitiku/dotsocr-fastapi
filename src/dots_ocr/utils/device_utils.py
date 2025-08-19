import torch
def pick_device_and_dtype():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        dtype = torch.bfloat16 if bf16_ok else torch.float16
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        dtype = torch.bfloat16
    else:
        dev = torch.device("cpu")
        dtype = torch.float32
    return dev, dtype

