# ml-models

Model quantization, conversion, and deployment for Capsule.

## Directory Structure

```
ml-models/
├── conversion/              # Export PyTorch → ONNX
│   ├── export_medasr_onnx.py      # Export google/medasr to FP32 ONNX (402 MB)
│   └── quantize_medasr_onnx.py    # Quantize FP32 → INT8 ONNX (101 MB, ↓75%)
│
├── quantization/            # Quantization scripts
│   ├── test_medasr.py             # MedASR inference + KenLM beam search + WER eval
│   └── load_medgemma_bnb4.py      # MedGemma BitsAndBytes 4-bit (GPU quantization step)
│
├── hf-models/               # HuggingFace model cards
│   ├── medasr-int8-onnx/          # Card for moisf56/medasr-conformer-ctc-int8-onnx
│   └── medgemma-4b-q3km-gguf/     # Card for moisf56/medgemma-4b-q3km-gguf
│
├── hf-spaces/               # HuggingFace Spaces demo
│   └── medasr-demo/               # Live demo: moisf56/capsule-medasr-demo
│
├── upload_to_hf.py          # Upload model files to HuggingFace Hub
└── upload_space_to_hf.py    # Deploy Gradio Space to HuggingFace
```

## Models

| Model | Base | Format | Size | Reduction |
|-------|------|--------|------|-----------|
| [moisf56/medasr-conformer-ctc-int8-onnx](https://huggingface.co/moisf56/medasr-conformer-ctc-int8-onnx) | google/medasr | ONNX INT8 | 101 MB | ↓75% |
| [moisf56/medgemma-4b-q3km-gguf](https://huggingface.co/moisf56/medgemma-4b-q3km-gguf) | google/medgemma-1.5-4b-it | GGUF Q3_K_M | 2.0 GB | ↓73% |

## Pipeline

```
google/medasr (402 MB PyTorch)
  └─► export_medasr_onnx.py    → medasr.onnx (402 MB FP32)
        └─► quantize_medasr_onnx.py → medasr_int8.onnx (101 MB INT8)
              └─► upload_to_hf.py → moisf56/medasr-conformer-ctc-int8-onnx

google/medgemma-1.5-4b-it (7.3 GB)
  └─► llama.cpp convert + Q3_K_M → medgemma-1.5-4b-it-Q3_K_M.gguf (2.0 GB)
        └─► upload_to_hf.py → moisf56/medgemma-4b-q3km-gguf
```

## MedASR Inference

The `test_medasr.py` script shows the full inference pipeline used in development:

```python
from ml-models.quantization.test_medasr import create_medasr_pipeline, transcribe

pipe = create_medasr_pipeline()           # loads google/medasr + KenLM 6-gram LM
text = transcribe("audio.wav", pipe)      # ~5% WER on medical speech
```

For on-device mobile (React Native), the ONNX model is used directly with manual
mel spectrogram computation — see `mobile/MedGemmaApp/src/MelSpectrogram.ts` and
`mobile/MedGemmaApp/src/CTCDecoder.ts`.

## MedGemma GGUF Conversion

```bash
# Requires llama.cpp
python llama.cpp/convert_hf_to_gguf.py google/medgemma-1.5-4b-it --outtype q3_K_M
```

## Upload to HuggingFace

```bash
HF_TOKEN=hf_xxx python ml-models/upload_to_hf.py
HF_TOKEN=hf_xxx python ml-models/upload_space_to_hf.py
```
