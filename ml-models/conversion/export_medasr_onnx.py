"""
Export MedASR to ONNX format.

MedASR is a 105M Conformer CTC model. We export just the neural network
(encoder + CTC head). The preprocessing (mel spectrogram) and postprocessing
(CTC decode) happen outside the ONNX model.

Pipeline:
  raw audio → [LasrFeatureExtractor] → mel spectrogram → [ONNX model] → logits → [CTC decode] → text

Usage:
  python export_medasr_onnx.py
"""

import os
import sys
import time
import numpy as np
import torch
import librosa
import huggingface_hub
from transformers import AutoModelForCTC, AutoProcessor

MODEL_ID = "google/medasr"
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "onnx")
ONNX_PATH = os.path.join(ONNX_DIR, "medasr.onnx")


def load_model():
    """Load MedASR model and processor."""
    print(f"Loading model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForCTC.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    print(f"Model type: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    return model, processor


def prepare_dummy_input(processor):
    """Create a real input from test audio for tracing."""
    audio_path = huggingface_hub.hf_hub_download(MODEL_ID, 'test_audio.wav')
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
    print(f"Input features shape: {inputs.input_features.shape}")
    if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
        print(f"Attention mask shape: {inputs.attention_mask.shape}")
    return inputs


def export_onnx(model, inputs):
    """Export model to ONNX using torch.onnx.export."""
    os.makedirs(ONNX_DIR, exist_ok=True)

    print(f"\nExporting to: {ONNX_PATH}")

    # Determine model inputs
    input_names = ["input_features"]
    forward_args = (inputs.input_features,)
    dynamic_axes = {
        "input_features": {0: "batch", 1: "time_steps", 2: "features"},
        "logits": {0: "batch", 1: "out_time"},
    }

    if hasattr(inputs, 'attention_mask') and inputs.attention_mask is not None:
        input_names.append("attention_mask")
        forward_args = (inputs.input_features, inputs.attention_mask)
        dynamic_axes["attention_mask"] = {0: "batch", 1: "time_steps"}

    start = time.time()
    torch.onnx.export(
        model,
        forward_args,
        ONNX_PATH,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        do_constant_folding=True,
    )
    elapsed = time.time() - start

    size_mb = os.path.getsize(ONNX_PATH) / 1024 / 1024
    print(f"Export completed in {elapsed:.1f}s")
    print(f"ONNX file size: {size_mb:.1f} MB")
    return ONNX_PATH


def verify_onnx(model, processor, onnx_path):
    """Verify ONNX model produces same output as PyTorch."""
    import onnxruntime as ort

    print("\n--- Verification ---")

    # Load test audio
    audio_path = huggingface_hub.hf_hub_download(MODEL_ID, 'test_audio.wav')
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")

    # PyTorch inference
    with torch.no_grad():
        pt_outputs = model(input_features=inputs.input_features)
        pt_logits = pt_outputs.logits.numpy()

    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    ort_inputs = {"input_features": inputs.input_features.numpy()}
    if "attention_mask" in [i.name for i in session.get_inputs()]:
        ort_inputs["attention_mask"] = inputs.attention_mask.numpy()

    ort_logits = session.run(None, ort_inputs)[0]

    # Compare
    print(f"PyTorch logits shape: {pt_logits.shape}")
    print(f"ONNX logits shape:    {ort_logits.shape}")

    max_diff = np.max(np.abs(pt_logits - ort_logits))
    mean_diff = np.mean(np.abs(pt_logits - ort_logits))
    print(f"Max absolute difference:  {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")

    # CTC greedy decode both
    pt_ids = np.argmax(pt_logits, axis=-1)[0]
    ort_ids = np.argmax(ort_logits, axis=-1)[0]
    match = np.array_equal(pt_ids, ort_ids)
    print(f"Token IDs match: {match}")

    if max_diff < 0.01:
        print("PASS: ONNX output matches PyTorch")
    else:
        print(f"WARNING: Large difference detected ({max_diff:.4f})")

    return max_diff


def test_onnx_transcription(processor, onnx_path):
    """Full transcription test with ONNX model."""
    import onnxruntime as ort

    print("\n--- Full Transcription Test ---")

    # Load audio
    audio_path = huggingface_hub.hf_hub_download(MODEL_ID, 'test_audio.wav')
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(speech, sampling_rate=sr, return_tensors="pt")

    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    ort_inputs = {"input_features": inputs.input_features.numpy()}
    if "attention_mask" in [i.name for i in session.get_inputs()]:
        ort_inputs["attention_mask"] = inputs.attention_mask.numpy()

    start = time.time()
    ort_logits = session.run(None, ort_inputs)[0]
    elapsed = time.time() - start
    print(f"ONNX inference time: {elapsed*1000:.0f}ms")

    # CTC greedy decode
    predicted_ids = np.argmax(ort_logits, axis=-1)
    transcription = processor.batch_decode(torch.tensor(predicted_ids))[0]

    print(f"Transcription: {transcription[:200]}...")
    return transcription


def main():
    print("=" * 60)
    print("MedASR ONNX Export")
    print("=" * 60)

    # Step 1: Load model
    model, processor = load_model()

    # Step 2: Prepare input
    inputs = prepare_dummy_input(processor)

    # Step 3: Export to ONNX
    onnx_path = export_onnx(model, inputs)

    # Step 4: Verify
    verify_onnx(model, processor, onnx_path)

    # Step 5: Test full transcription
    test_onnx_transcription(processor, onnx_path)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print(f"  1. Quantize: python quantize_medasr_onnx.py")
    print(f"  2. ONNX file: {onnx_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
