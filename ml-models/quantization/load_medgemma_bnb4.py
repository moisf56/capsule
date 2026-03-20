"""
MedGemma 1.5-4B BitsAndBytes 4-bit Quantization Script

This script loads the official MedGemma model with 4-bit quantization
using BitsAndBytes NF4 (NormalFloat4) - the recommended quantization
type for LLMs.

Requirements:
- Accept license at: https://huggingface.co/google/medgemma-1.5-4b-it
- Login: huggingface-cli login
- GPU with ~8GB+ VRAM (or will use CPU offloading)

Usage:
    python load_medgemma_bnb4.py
"""

import os
import gc

# Set memory optimization BEFORE importing torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

import torch
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
)


def clear_memory():
    """Clear GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_quantization_config():
    """
    Create optimal BitsAndBytes 4-bit quantization config.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16 for lower memory
        bnb_4bit_use_double_quant=True,
    )


def load_medgemma_quantized(model_id: str = "google/medgemma-1.5-4b-it"):
    """
    Load MedGemma with 4-bit quantization with aggressive memory management.
    """
    print(f"Loading {model_id} with BNB 4-bit quantization...")
    print("Using aggressive memory optimization for 8GB GPU...")

    clear_memory()

    quantization_config = get_quantization_config()

    # Very aggressive memory limits - offload most to CPU
    max_memory = {
        0: "5GiB",  # Only use 5GB of GPU
        "cpu": "24GiB"
    }

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_folder="offload_weights",  # Disk offloading if needed
        )
    except torch.cuda.OutOfMemoryError:
        print("\n⚠️  Still OOM with 5GB limit. Trying with 4GB...")
        clear_memory()

        max_memory[0] = "4GiB"
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            max_memory=max_memory,
            low_cpu_mem_usage=True,
            offload_folder="offload_weights",
        )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"\n✅ Model loaded successfully!")

    # Print memory usage
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

    return model, processor


def test_inference(model, processor):
    """Test with a simple medical text prompt."""
    print("\n" + "="*50)
    print("Testing inference with medical prompt...")
    print("="*50)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "A 58-year-old male presents with chest pain radiating to the left arm. Generate a brief assessment."
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Move to appropriate device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

    input_len = inputs["input_ids"].shape[-1]
    print(f"Input tokens: {input_len}")
    print("Generating response...")

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,  # Reduced for memory
            do_sample=False,
        )

    generated_tokens = outputs[0][input_len:]
    response = processor.decode(generated_tokens, skip_special_tokens=True)

    print(f"\nGenerated ({len(generated_tokens)} tokens):")
    print("-" * 50)
    print(response)
    print("-" * 50)

    return response


def main():
    """Main entry point."""
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. BitsAndBytes 4-bit requires GPU.")
        return

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM: {vram:.1f}GB")

    if vram < 10:
        print(f"⚠️  Limited VRAM detected. Will use aggressive CPU offloading.")
    print()

    # Clear any existing allocations
    clear_memory()

    # Load model
    model, processor = load_medgemma_quantized()

    # Test inference
    test_inference(model, processor)

    print("\n✅ MedGemma BNB 4-bit setup complete!")


if __name__ == "__main__":
    main()
