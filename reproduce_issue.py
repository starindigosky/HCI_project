import torch
from transformers import VitsModel, VitsTokenizer
import os

# Use cache dir from existing project to avoid re-downloading if possible
CACHE_DIR = "./model_cache"
MODEL_NAME = "facebook/mms-tts-nan"


def test_tokenization():
    print(f"Loading model {MODEL_NAME}...")
    try:
        tokenizer = VitsTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        model = VitsModel.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    text_hanzi = "汝好"
    print(f"\nTesting Hanzi input: '{text_hanzi}'")
    inputs = tokenizer(text_hanzi, return_tensors="pt")
    print(f"Token IDs: {inputs.input_ids}")
    print(f"Input shape: {inputs.input_ids.shape}")

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            print("Inference successful!")
            print(f"Output waveform shape: {outputs.waveform.shape}")
    except Exception as e:
        print(f"Inference failed: {e}")

    text_poj = "Lí hó"
    print(f"\nTesting POJ input: '{text_poj}'")
    inputs = tokenizer(text_poj, return_tensors="pt")
    print(f"Token IDs: {inputs.input_ids}")
    print(f"Input shape: {inputs.input_ids.shape}")

    try:
        with torch.no_grad():
            outputs = model(**inputs)
            print("Inference successful!")
            print(f"Output waveform shape: {outputs.waveform.shape}")
    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    test_tokenization()
