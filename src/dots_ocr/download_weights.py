from argparse import ArgumentParser
import os
from huggingface_hub import snapshot_download

import re, json, shutil
from pathlib import Path

# Adopted from https://github.com/rednote-hilab/dots.ocr/blob/master/tools/download_model.py

def main():
    type = "huggingface"
    model_id = "rednote-hilab/dots.ocr"
    model_dir = "weights/DotsOCR"
    attention_impl = os.environ.get("ATTN_IMPL", "sdpa")  # default to sdpa
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    try:
        snapshot_download(repo_id=model_id, local_dir=model_dir, local_dir_use_symlinks=False, resume_download=True)
        # The DotsOCR currently doesn't have __init__.py. 
        print(f"model downloaded to {model_dir}")
    except KeyboardInterrupt:
        print("Download interrupted. Exiting...")

    config_path = os.path.join(model_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        config['vision_config']["attn_implementation"] = attention_impl
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Updated attention implementation in {config_path} to {attention_impl}")


if __name__ == "__main__":
    main()
