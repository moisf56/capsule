---
title: Capsule — Medical Speech Recognition
emoji: 🎙️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.9.0"
app_file: app.py
pinned: true
license: apache-2.0
models:
  - moisf56/medasr-conformer-ctc-int8-onnx
  - moisf56/medgemma-4b-q3km-gguf
---

# Capsule — Medical Speech Recognition

Interactive demo for the [Capsule](https://github.com/moisf56/capsule) project —
submitted to the **MedGemma Impact Challenge (Edge AI Prize)**.

**Models used:**
- [`moisf56/medasr-conformer-ctc-int8-onnx`](https://huggingface.co/moisf56/medasr-conformer-ctc-int8-onnx) — 101 MB INT8 ONNX
- [`moisf56/medgemma-4b-q3km-gguf`](https://huggingface.co/moisf56/medgemma-4b-q3km-gguf) — 2.0 GB GGUF Q3_K_M
