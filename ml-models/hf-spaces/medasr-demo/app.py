"""
Capsule — MedASR + MedGemma Demo
Gradio Space: moisf56/capsule-medasr-demo

Pipeline:
  Audio → LasrFeatureExtractor → google/medasr (Conformer CTC)
        → Manual CTC greedy decode → Medical Text Formatting
        → (optional) MedGemma 4B Q3_K_M → SOAP Note
"""

import re
import threading

import gradio as gr
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import torch
import transformers
from huggingface_hub import hf_hub_download

matplotlib.use("Agg")

# ── MedASR: direct model inference (pipeline CTC collapse is broken for Lasr) ─

MODEL_ID = "google/medasr"
print(f"Loading MedASR ({MODEL_ID})...")
_fe    = transformers.LasrFeatureExtractor.from_pretrained(MODEL_ID)
_model = transformers.AutoModelForCTC.from_pretrained(MODEL_ID)
_tok   = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
_model.eval()
print("MedASR ready.")

# ── MedGemma: lazy load ───────────────────────────────────────────────────────

_llm      = None
_llm_lock = threading.Lock()


def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    with _llm_lock:
        if _llm is not None:
            return _llm
        from llama_cpp import Llama
        print("Downloading MedGemma Q3_K_M (2 GB) — first SOAP request only...")
        gguf = hf_hub_download(
            "moisf56/medgemma-4b-q3km-gguf",
            "medgemma-1.5-4b-it-Q3_K_M.gguf",
        )
        print("Loading MedGemma into memory...")
        _llm = Llama(
            model_path=gguf,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False,
        )
        print("MedGemma ready.")
        return _llm


# ── Audio parameters (mel visualisation) ─────────────────────────────────────

SR         = 16_000
N_FFT      = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS     = 128

_HANN = scipy.signal.windows.hann(WIN_LENGTH, sym=True).astype(np.float64)
_MEL_FILTERS = librosa.filters.mel(
    sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=SR / 2
).astype(np.float64)


def compute_mel(audio: np.ndarray) -> np.ndarray:
    n_frames = (len(audio) - WIN_LENGTH) // HOP_LENGTH + 1
    mel_spec = np.empty((N_MELS, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame    = audio[i * HOP_LENGTH : i * HOP_LENGTH + WIN_LENGTH].astype(np.float64)
        windowed = frame * _HANN
        padded   = np.zeros(N_FFT, dtype=np.float64)
        padded[:WIN_LENGTH] = windowed
        fft   = np.fft.rfft(padded)
        power = fft.real ** 2 + fft.imag ** 2
        mel   = _MEL_FILTERS @ power
        mel_spec[:, i] = np.log(np.maximum(mel, 1e-5)).astype(np.float32)
    return mel_spec


# ── CTC decode ────────────────────────────────────────────────────────────────

def ctc_greedy_decode(logits_tensor: torch.Tensor) -> str:
    """Argmax → collapse consecutive duplicates → remove blank (token 0) → decode."""
    ids = logits_tensor.argmax(dim=-1).tolist()   # (T,)
    collapsed, prev = [], -1
    for id_ in ids:
        if id_ != 0 and id_ != prev:
            collapsed.append(id_)
        prev = id_
    return _tok.decode(collapsed) if collapsed else ""


# ── Text formatting ───────────────────────────────────────────────────────────

def format_transcription(text: str) -> str:
    text = re.sub(r"\[([A-Z\s]+)\]", lambda m: m.group(1).title(), text)
    for tok in ["</s>", "<s>", "<epsilon>", "<pad>"]:
        text = text.replace(tok, "")
    for tok, char in [
        ("{period}", "."), ("{comma}", ","), ("{colon}", ":"),
        ("{semicolon}", ";"), ("{new paragraph}", "\n\n"),
        ("{question mark}", "?"), ("{exclamation point}", "!"), ("{hyphen}", "-"),
    ]:
        text = text.replace(tok, char)
    text = re.sub(r"\{[^}]*\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([.,:;?!\-])", r"\1", text)
    return text


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_mel(mel: np.ndarray, duration_s: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    img = ax.imshow(
        mel, aspect="auto", origin="lower", cmap="magma",
        extent=[0, duration_s, 0, SR / 2 / 1000],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)", color="white", fontsize=10)
    ax.set_ylabel("Frequency (kHz)", color="white", fontsize=10)
    ax.set_title(
        f"Log Mel Spectrogram  ·  {mel.shape[1]} frames  ·  128 mel bins  ·  512-pt FFT",
        color="white", fontsize=11, pad=8,
    )
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Log Energy", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    plt.tight_layout()
    return fig


# ── SOAP prompt ───────────────────────────────────────────────────────────────

SOAP_SYSTEM = (
    "You are a clinical documentation assistant. "
    "Generate a structured SOAP note from the provided medical dictation transcript. "
    "Use standard SOAP format with clear section headers: "
    "Subjective, Objective, Assessment, Plan. "
    "Be concise, clinical, and accurate."
)

SOAP_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{system}\n\n"
    "Transcript:\n{transcript}\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


# ── Inference ─────────────────────────────────────────────────────────────────

def transcribe(audio_path):
    if audio_path is None:
        return None, "", ""

    audio_f32, _ = librosa.load(audio_path, sr=SR, mono=True)
    duration_s   = len(audio_f32) / SR
    mel          = compute_mel(audio_f32)

    # LasrFeatureExtractor → model → manual CTC greedy decode
    inputs  = _fe(audio_f32, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits[0]   # (T, vocab)

    raw_text  = ctc_greedy_decode(logits)
    formatted = format_transcription(raw_text)

    print(f"[ASR] raw:       {raw_text[:200]}")
    print(f"[ASR] formatted: {formatted[:200]}")

    return plot_mel(mel, duration_s), formatted, raw_text


def generate_soap(transcript: str):
    transcript = transcript.strip()
    if not transcript:
        return "⚠️ Transcribe audio first, then click Generate SOAP Note."
    try:
        llm = get_llm()
    except Exception as e:
        return f"❌ Failed to load MedGemma: {e}"

    prompt = SOAP_PROMPT_TEMPLATE.format(system=SOAP_SYSTEM, transcript=transcript)
    try:
        out = llm(
            prompt,
            max_tokens=768,
            temperature=0.3,
            repeat_penalty=1.1,
            stop=["<end_of_turn>"],
        )
        return out["choices"][0]["text"].strip()
    except Exception as e:
        return f"❌ Generation error: {e}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

DESCRIPTION = """
## 🎙️ Capsule — MedASR + MedGemma On-Device Pipeline

**MedASR:** 105M Conformer CTC · `google/medasr` · ~5% WER on medical speech
**MedGemma:** 4B multimodal · GGUF Q3_K_M · **2.0 GB** (↓73% from 7.3 GB)

Dictate a clinical encounter → get a transcript → generate a structured SOAP note.
This is the exact pipeline running inside **[Capsule](https://github.com/mo-saif/capsule)**,
tested on a $150 Android phone (Tecno Spark 40 · 8 GB RAM · CPU only — no GPU anywhere).

> **Note:** SOAP generation loads MedGemma (2 GB) on first click — may take ~60 s.

*Part of the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) ·
Models: [MedASR](https://huggingface.co/moisf56/medasr-conformer-ctc-int8-onnx) · [MedGemma GGUF](https://huggingface.co/moisf56/medgemma-4b-q3km-gguf)*
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Capsule — MedASR + MedGemma") as demo:
    gr.Markdown(DESCRIPTION)

    gr.Markdown("### Step 1 — Record or upload audio")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Clinical audio (any sample rate · mono or stereo)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            transcribe_btn = gr.Button("▶  Transcribe with MedASR", variant="primary")
        with gr.Column(scale=2):
            mel_plot = gr.Plot(label="Log Mel Spectrogram")

    gr.Markdown("### Step 2 — Review and edit transcript")
    gr.Markdown("_MedASR output below. Correct any errors before generating the SOAP note._")
    formatted_output = gr.Textbox(
        label="Transcript  ✏️ editable",
        lines=5,
        show_copy_button=True,
        interactive=True,
        placeholder="Transcript will appear here after Step 1...",
    )

    with gr.Accordion("Raw ASR output (before formatting)", open=False):
        raw_output = gr.Textbox(
            label="Raw MedASR text  (special tokens visible)",
            lines=3,
            show_copy_button=True,
        )

    gr.Markdown("### Step 3 — Generate SOAP Note with MedGemma 4B")
    gr.Markdown(
        "_Generates from the transcript above — edits you made in Step 2 are included._  "
        "_First click loads MedGemma (2 GB) — expect ~60 s wait._"
    )
    soap_btn  = gr.Button("▶  Generate SOAP Note", variant="secondary")
    soap_note = gr.Textbox(
        label="SOAP Note  (MedGemma 4B Q3_K_M)",
        lines=14,
        show_copy_button=True,
        placeholder="SOAP note will appear here after Step 3...",
    )

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[mel_plot, formatted_output, raw_output],
        api_name=False,
    )
    soap_btn.click(
        fn=generate_soap,
        inputs=formatted_output,
        outputs=soap_note,
        api_name=False,
    )

    gr.Markdown("""
---
**Pipeline details**
| Step | Detail |
|------|--------|
| Mel spectrogram | 512-pt FFT · 128 mel bins · symmetric Hann window · log(clamp(·, 1e-5)) |
| ASR | `google/medasr` Conformer CTC · LasrFeatureExtractor · CTC greedy decode · ~5% WER |
| Medical formatting | `{period}` → `.` · `{comma}` → `,` · `[EXAM TYPE]` → section headers |
| SOAP generation | MedGemma 4B Q3_K_M · 768 max tokens · temp 0.3 · repeat penalty 1.1 |
| On-device (mobile) | MedASR: ONNX INT8 (101 MB, ↓75%) · MedGemma: llama.cpp Q3_K_M (2.0 GB, ↓73%) |
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False, ssr_mode=False)
