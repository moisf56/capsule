---
base_model: google/medasr
base_model_relation: quantized
language:
  - en
license: apache-2.0
tags:
  - medical
  - speech-recognition
  - conformer-ctc
  - onnx
  - int8
  - quantized
  - on-device
  - mobile
  - android
pipeline_tag: automatic-speech-recognition
---

# MedASR — Conformer CTC INT8 ONNX (101 MB)

> **Attribution:** This is a quantized derivative of [`google/medasr`](https://huggingface.co/google/medasr) (Apache 2.0). All credit for the base model goes to Google.

Quantized by [Mohammed K. A. Abed](https://huggingface.co/moisf56) as part of the **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge)** — optimised for on-device medical speech recognition on mobile and CPU-only hardware.

| | |
|---|---|
| **Base model** | `google/medasr` (105M parameters, Conformer CTC) |
| **Original size** | 402 MB |
| **Quantized size** | 101 MB |
| **Reduction** | 75% |
| **Method** | ONNX INT8 dynamic quantization |
| **Runtime** | ONNX Runtime (CPU) |
| **Tested on** | Tecno Spark 40 · MediaTek Helio G100 · 8 GB RAM |

---

## About this quantization

Dynamic INT8 quantization was applied to the full Conformer CTC graph using ONNX Runtime's `quantize_dynamic`. All matrix multiplications are quantized to INT8; activations remain in FP32. The result fits in the assets bundle of a React Native Android app and loads in under 3 seconds on a mid-range phone.

## Audio pipeline

The model expects a mel spectrogram computed with the following settings (matching `LasrFeatureExtractor`):

| Parameter | Value |
|-----------|-------|
| Sample rate | 16 000 Hz, mono |
| FFT size | 512 |
| Mel bins | 128 |
| Window | Hann |
| Hop length | 160 samples (10 ms) |
| Frequency range | 0–8 000 Hz |

CTC greedy decoding is performed over a 512-token SentencePiece vocabulary (blank = token 0). Post-processing converts special punctuation tokens (`{period}`, `{comma}`, etc.) and section-header tokens (`[EXAM TYPE]`) into formatted medical text.

## Files

| File | Description |
|------|-------------|
| `medasr_int8.onnx` | INT8 quantized ONNX model |
| `medasr_vocab.json` | 512-token SentencePiece vocabulary |

## Usage

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np
import json

# Load vocab and model
with open("medasr_vocab.json") as f:
    vocab = json.load(f)

session = ort.InferenceSession(
    "medasr_int8.onnx",
    providers=["CPUExecutionProvider"]
)

# audio_array: float32 numpy array, shape (T,), resampled to 16 kHz mono
# mel: shape (1, 128, T') — compute with your mel spectrogram library
mel = compute_mel_spectrogram(audio_array)   # see audio pipeline table above

logits = session.run(None, {"input": mel})[0]   # (1, T', 512)
tokens = np.argmax(logits[0], axis=-1)

# CTC greedy decode (remove blanks and repeated tokens)
decoded = []
prev = -1
for t in tokens:
    if t != 0 and t != prev:
        decoded.append(vocab[t])
    prev = t

transcript = " ".join(decoded)
```

### React Native / Android (onnxruntime-react-native)

This is how the model is used inside [Capsule](https://github.com/mo-saif/capsule):

```typescript
import { InferenceSession, Tensor } from 'onnxruntime-react-native';

const session = await InferenceSession.create('file:///data/local/tmp/medasr_int8.onnx');

const feeds = { input: new Tensor('float32', melData, [1, 128, timeSteps]) };
const results = await session.run(feeds);
const logits = results['output'].data as Float32Array;
```

See [`CTCDecoder.ts`](https://github.com/mo-saif/capsule/blob/main/mobile/MedGemmaApp/src/CTCDecoder.ts) and [`MelSpectrogram.ts`](https://github.com/mo-saif/capsule/blob/main/mobile/MedGemmaApp/src/MelSpectrogram.ts) for the full reference implementation.

---

## Performance

Measured on **Tecno Spark 40** (MediaTek Helio G100, 8 GB RAM, CPU-only):

| Metric | Value |
|--------|-------|
| Transcription latency (30 s audio) | < 10 s |
| Mel spectrogram computation | 3–5 s |
| Peak memory | ~300 MB |
| Battery impact | < 3% per hour of active use |

---

## Quantization details

```bash
python -m onnxruntime.quantization.quantize \
    --model medasr.onnx \
    --output medasr_int8.onnx \
    --quant_type dynamic \
    --weight_type QInt8
```

---

## License

Weights inherit the license of the base model [`google/medasr`](https://huggingface.co/google/medasr). The quantization artifacts (this card, vocab file) are released under Apache 2.0.

---

## Citation

If you use this model, please cite the original MedASR work and acknowledge the Capsule project:

```
@misc{capsule2026,
  title  = {Capsule: Edge AI Clinical Documentation with Agentic Intelligence},
  author = {Mohammed K. A. Abed},
  year   = {2026},
  url    = {https://github.com/mo-saif/capsule}
}
```
