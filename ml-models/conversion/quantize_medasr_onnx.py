"""
Quantize MedASR ONNX model to INT8 and test the full pipeline.

Includes proper CTC greedy decoding (not using processor.batch_decode,
which expects generate() output, not raw logits).

Usage:
  python quantize_medasr_onnx.py
"""

import os
import sys
import time
import numpy as np
import torch
import librosa
import huggingface_hub

# Add parent for test_medasr imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantization"))
from test_medasr import format_transcription, calculate_wer

MODEL_ID = "google/medasr"
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "onnx")
FP32_PATH = os.path.join(ONNX_DIR, "medasr.onnx")
INT8_PATH = os.path.join(ONNX_DIR, "medasr_int8.onnx")


def ctc_greedy_decode(logits, tokenizer):
    """
    CTC greedy decoding: argmax per frame, collapse repeats, remove blanks.

    CTC (Connectionist Temporal Classification) outputs one token per time frame.
    Many frames map to the same token (repeats) or to blank (token 0).
    Greedy decode: take argmax, collapse consecutive duplicates, remove blanks.
    """
    predicted_ids = np.argmax(logits[0], axis=-1)  # [time_steps]

    # Collapse repeats and remove blank (id=0)
    tokens = []
    prev_id = -1
    for token_id in predicted_ids:
        if token_id != prev_id and token_id != 0:  # not repeat, not blank
            tokens.append(int(token_id))
        prev_id = token_id

    # Decode tokens to text using tokenizer
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text


def quantize():
    """Quantize FP32 ONNX to INT8."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if not os.path.exists(FP32_PATH):
        print(f"ERROR: FP32 model not found at {FP32_PATH}")
        print("Run export_medasr_onnx.py first.")
        return False

    fp32_size = os.path.getsize(FP32_PATH) / 1024 / 1024
    print(f"FP32 model: {fp32_size:.1f} MB")

    print("Quantizing to INT8...")
    start = time.time()
    quantize_dynamic(
        FP32_PATH,
        INT8_PATH,
        weight_type=QuantType.QInt8,
    )
    elapsed = time.time() - start

    int8_size = os.path.getsize(INT8_PATH) / 1024 / 1024
    reduction = (1 - int8_size / fp32_size) * 100
    print(f"INT8 model: {int8_size:.1f} MB ({reduction:.0f}% reduction)")
    print(f"Quantization time: {elapsed:.1f}s")
    return True


def test_pipeline(onnx_path, label=""):
    """Test full transcription pipeline with an ONNX model."""
    import onnxruntime as ort
    from transformers import AutoProcessor, AutoTokenizer

    print(f"\n--- Testing {label} ---")
    print(f"Model: {onnx_path}")
    print(f"Size: {os.path.getsize(onnx_path) / 1024 / 1024:.1f} MB")

    # Load processor and tokenizer
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)

    # Load test audio
    audio_path = huggingface_hub.hf_hub_download(MODEL_ID, 'test_audio.wav',
                                                  local_files_only=True)
    speech, sr = librosa.load(audio_path, sr=16000)
    print(f"Audio: {len(speech)/sr:.1f}s at {sr}Hz")

    # Preprocess: audio → mel spectrogram
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")

    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    ort_inputs = {"input_features": inputs.input_features.numpy()}
    if "attention_mask" in [i.name for i in session.get_inputs()]:
        ort_inputs["attention_mask"] = inputs.attention_mask.numpy()

    start = time.time()
    logits = session.run(None, ort_inputs)[0]
    inference_ms = (time.time() - start) * 1000
    print(f"Inference: {inference_ms:.0f}ms")
    print(f"Logits shape: {logits.shape}")

    # CTC greedy decode
    transcription_raw = ctc_greedy_decode(logits, tokenizer)
    transcription = format_transcription(transcription_raw)

    print(f"\nTranscription:")
    print(transcription)

    # Calculate WER against reference
    reference = (
        "Exam type CT chest PE protocol. "
        "Indication 54 year old female, shortness of breath, evaluate for PE. "
        "Technique standard protocol. "
        "Findings: Pulmonary vasculature: The main PA is patent. "
        "There are filling defects in the segmental branches of the right lower lobe, "
        "compatible with acute PE. No saddle embolus. "
        "Lungs: No pneumothorax. Small bilateral effusions, right greater than left. "
        "Impression: Acute segmental PE, right lower lobe."
    )
    wer = calculate_wer(reference, transcription)
    print(f"\nWER: {wer:.1%}")

    return wer, inference_ms


def main():
    print("=" * 60)
    print("MedASR ONNX Quantization + Full Pipeline Test")
    print("=" * 60)

    # Step 1: Quantize
    if not quantize():
        return

    # Step 2: Test FP32
    wer_fp32, time_fp32 = test_pipeline(FP32_PATH, "FP32 ONNX")

    # Step 3: Test INT8
    wer_int8, time_int8 = test_pipeline(INT8_PATH, "INT8 ONNX (Quantized)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    fp32_size = os.path.getsize(FP32_PATH) / 1024 / 1024
    int8_size = os.path.getsize(INT8_PATH) / 1024 / 1024
    print(f"{'':20} {'FP32':>10} {'INT8':>10}")
    print(f"{'Size (MB)':20} {fp32_size:>10.1f} {int8_size:>10.1f}")
    print(f"{'Inference (ms)':20} {time_fp32:>10.0f} {time_int8:>10.0f}")
    print(f"{'WER':20} {wer_fp32:>9.1%} {wer_int8:>9.1%}")
    print(f"{'Size reduction':20} {'--':>10} {(1-int8_size/fp32_size)*100:>9.0f}%")
    print("=" * 60)

    if wer_int8 <= wer_fp32 + 0.02:  # Allow 2% WER degradation
        print("PASS: INT8 quality is acceptable")
    else:
        print(f"WARNING: INT8 WER degraded by {(wer_int8-wer_fp32)*100:.1f}%")


if __name__ == "__main__":
    main()
