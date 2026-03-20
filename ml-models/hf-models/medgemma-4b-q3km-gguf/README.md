---
base_model: google/medgemma-1.5-4b-it
base_model_relation: quantized
language:
  - en
license: gemma
tags:
  - medical
  - clinical
  - gguf
  - q3_k_m
  - quantized
  - on-device
  - mobile
  - android
  - llama-cpp
  - soap-notes
  - clinical-nlp
pipeline_tag: text-generation
---

# MedGemma 4B — Q3\_K\_M GGUF (2.0 GB)

> **License notice:** This is a quantized derivative of [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it) and is governed by the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). You must accept those terms on the original model page before downloading or using this file. All credit for the base model goes to Google.

Quantized by [Mohammed K. A. Abed](https://huggingface.co/moisf56) as part of the **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge)** — optimised for on-device clinical inference on mobile and CPU-only hardware.

| | |
|---|---|
| **Base model** | `google/medgemma-1.5-4b-it` (4B parameters) |
| **Original size** | 7.3 GB (BF16) |
| **Quantized size** | 2.0 GB |
| **Reduction** | 73% |
| **Method** | GGUF Q3\_K\_M (3-bit k-means quantization via llama.cpp) |
| **Runtime** | llama.cpp / llama.rn |
| **Tested on** | Tecno Spark 40 · MediaTek Helio G100 · 8 GB RAM |

---

## About this quantization

Q3\_K\_M uses 3-bit k-means quantization with medium-sized super-blocks. Compared to lower bit-widths:

| Variant | Bits | Size | Notes |
|---------|------|------|-------|
| Q4\_K\_M | 4.83 | 2.4 GB | Higher quality — recommended for workstation use |
| **Q3\_K\_M** | **3.07** | **2.0 GB** | **Best fit for 8 GB mobile RAM — used in Capsule** |
| Q2\_K | 2.96 | 1.5 GB | Perplexity penalty too high (+3.5 PPL) |
| IQ2\_M | 2.7 | 1.3 GB | Too slow on mobile CPU |

Q3\_K\_M was selected after benchmarking on a mid-range Android phone (Tecno Spark 40, MediaTek Helio G100). It is the highest quality variant that fits within the phone's working RAM budget after the OS and app overhead are accounted for.

## Files

| File | Description |
|------|-------------|
| `medgemma-1.5-4b-it-Q3_K_M.gguf` | Q3\_K\_M quantized GGUF weights |

## Usage

### llama.cpp CLI

```bash
./llama-cli \
    -m medgemma-1.5-4b-it-Q3_K_M.gguf \
    -n 512 \
    --ctx-size 4096 \
    --temp 0.3 \
    --repeat-penalty 1.1 \
    -p "<start_of_turn>user\nGenerate a SOAP note for the following transcript:\n\n{transcript}<end_of_turn>\n<start_of_turn>model\n"
```

### llama.cpp server (workstation)

```bash
./llama-server \
    -m medgemma-1.5-4b-it-Q3_K_M.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    --ctx-size 4096 \
    --n-predict 512
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="medgemma-1.5-4b-it-Q3_K_M.gguf",
    n_ctx=4096,
    n_threads=8,
)

prompt = (
    "<start_of_turn>user\n"
    "Generate a SOAP note for the following transcript:\n\n"
    "{transcript}"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

output = llm(
    prompt,
    max_tokens=512,
    temperature=0.3,
    repeat_penalty=1.1,
    stop=["<end_of_turn>"],
)
print(output["choices"][0]["text"])
```

### React Native / Android (llama.rn)

This is how the model is used inside [Capsule](https://github.com/mo-saif/capsule):

```typescript
import { initLlama, LlamaContext } from 'llama.rn';

const context: LlamaContext = await initLlama({
  model: '/data/local/tmp/medgemma.gguf',
  n_ctx: 4096,
  n_threads: 4,
});

const result = await context.completion({
  prompt: `<start_of_turn>user\n${systemPrompt}\n\n${transcript}<end_of_turn>\n<start_of_turn>model\n`,
  n_predict: 512,
  temperature: 0.3,
  repeat_penalty: 1.1,
  stop: ['<end_of_turn>'],
});
```

See [`App.tsx`](https://github.com/mo-saif/capsule/blob/main/mobile/MedGemmaApp/App.tsx) for the full reference implementation including memory management (sequential loading with MedASR).

---

## Clinical tasks demonstrated

| Task | Where |
|------|-------|
| SOAP note generation from dictation transcript | On-device (phone) |
| Lab result summarisation and interpretation | On-device (phone) |
| Clinical chat with voice input | On-device (phone) |
| Agentic SOAP enhancement (DDI · ICD-10 · lab correlation) | Workstation (Q4\_K\_M) |
| EHR Navigator — natural-language FHIR queries | Workstation (Q4\_K\_M) |
| Radiology report generation (8 imaging modalities) | Workstation + mmproj |

---

## Performance

Measured on **Tecno Spark 40** (MediaTek Helio G100, 8 GB RAM, CPU-only):

| Metric | Value |
|--------|-------|
| SOAP note generation | ~60 s |
| Peak memory during inference | ~3.2 GB |
| Battery impact | < 3% per hour of active use |
| Model load time | ~8 s (pre-loaded during transcript review) |

No GPU required. Also tested on Ryzen 7 8845HS (32 GB RAM) where the Q4\_K\_M variant is used for the workstation pipeline.

---

## Quantization command

```bash
# Convert base model weights to GGUF (run inside llama.cpp repo)
python convert_hf_to_gguf.py \
    /path/to/medgemma-1.5-4b-it \
    --outfile medgemma-1.5-4b-it-f16.gguf \
    --outtype f16

# Quantize to Q3_K_M
./llama-quantize \
    medgemma-1.5-4b-it-f16.gguf \
    medgemma-1.5-4b-it-Q3_K_M.gguf \
    Q3_K_M
```

---

## Memory layout on device

MedASR (101 MB ONNX) and MedGemma (2.0 GB GGUF) are loaded sequentially — never simultaneously — to stay within 8 GB RAM:

```
Record dictation  →  Load MedASR (101 MB)  →  Transcribe  →  Unload MedASR
                  →  Pre-load MedGemma during transcript review (2.0 GB)
                  →  Generate SOAP note  →  Unload MedGemma (on demand)
```

Total on-device footprint: **2.1 GB** active at any one time.

---

## License

This model is derived from `google/medgemma-1.5-4b-it` and inherits the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). You must accept those terms before downloading or using this model.

---

## Citation

If you use this model, please cite the original MedGemma work and acknowledge the Capsule project:

```
@misc{capsule2026,
  title  = {Capsule: Edge AI Clinical Documentation with Agentic Intelligence},
  author = {Mohammed K. A. Abed},
  year   = {2026},
  url    = {https://github.com/mo-saif/capsule}
}
```
