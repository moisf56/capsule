"""
Capsule — MedASR + MedGemma Demo
Gradio Space: moisf56/capsule-medasr-demo

Pipeline:
  Audio → Resample 16kHz → Log Mel Spectrogram (LasrFeatureExtractor)
        → Conformer CTC INT8 ONNX → Greedy Decode → Medical Text Formatting
        → (optional) MedGemma 4B Q3_K_M → SOAP Note
"""

import json
import re
import threading

import gradio as gr
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import scipy.signal
from huggingface_hub import hf_hub_download

matplotlib.use("Agg")

# ── MedASR: model & vocab ─────────────────────────────────────────────────────

print("Loading MedASR (INT8 ONNX)...")
_asr_model  = hf_hub_download("moisf56/medasr-conformer-ctc-int8-onnx", "medasr_int8.onnx")
_vocab_file = hf_hub_download("moisf56/medasr-conformer-ctc-int8-onnx", "medasr_vocab.json")

with open(_vocab_file) as f:
    VOCAB: list[str] = json.load(f)

ASR_SESSION = ort.InferenceSession(_asr_model, providers=["CPUExecutionProvider"])
ASR_INPUT   = ASR_SESSION.get_inputs()[0].name
ASR_OUTPUT  = ASR_SESSION.get_outputs()[0].name
print(f"MedASR ready  ({ASR_INPUT} → {ASR_OUTPUT})")

# ── MedGemma: lazy load ───────────────────────────────────────────────────────

_llm       = None
_llm_lock  = threading.Lock()

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

# ── Audio parameters (matching LasrFeatureExtractor exactly) ──────────────────

SR         = 16_000
N_FFT      = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS     = 128
N_FREQ     = N_FFT // 2 + 1   # 257

# Symmetric Hann window — matches PyTorch hann_window(periodic=False)
_HANN = scipy.signal.windows.hann(WIN_LENGTH, sym=True).astype(np.float64)

# Mel filterbank (same parameters as LasrFeatureExtractor)
_MEL_FILTERS = librosa.filters.mel(
    sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=0.0, fmax=SR / 2
).astype(np.float64)   # (128, 257)

# ── DSP ───────────────────────────────────────────────────────────────────────

def compute_mel(audio: np.ndarray) -> np.ndarray:
    """
    Log mel spectrogram matching LasrFeatureExtractor._torch_extract_fbank_features.
      1. Frame with WIN_LENGTH / HOP_LENGTH
      2. Symmetric Hann window (periodic=False)
      3. Real FFT (N_FFT=512) → power spectrum
      4. Mel filterbank → log(clamp(·, 1e-5))
    Returns (N_MELS, T) float32.
    """
    n_frames = (len(audio) - WIN_LENGTH) // HOP_LENGTH + 1
    mel_spec = np.empty((N_MELS, n_frames), dtype=np.float32)

    for i in range(n_frames):
        frame = audio[i * HOP_LENGTH : i * HOP_LENGTH + WIN_LENGTH].astype(np.float64)
        windowed = frame * _HANN
        padded = np.zeros(N_FFT, dtype=np.float64)
        padded[:WIN_LENGTH] = windowed

        fft   = np.fft.rfft(padded)
        power = fft.real ** 2 + fft.imag ** 2     # (257,)
        mel   = _MEL_FILTERS @ power               # (128,)
        mel_spec[:, i] = np.log(np.maximum(mel, 1e-5)).astype(np.float32)

    return mel_spec   # (128, T)


# ── CTC decoder ───────────────────────────────────────────────────────────────

def ctc_greedy_decode(logits: np.ndarray) -> tuple[list[str], str]:
    """Greedy CTC: argmax → collapse duplicates → remove blank (token 0)."""
    tokens: list[str] = []
    prev_id = -1
    for t in range(logits.shape[0]):
        max_id = int(np.argmax(logits[t]))
        if max_id != 0 and max_id != prev_id:
            if max_id < len(VOCAB):
                tokens.append(VOCAB[max_id])
        prev_id = max_id
    return tokens, "".join(tokens)


def format_transcription(text: str) -> str:
    """Convert MedASR special tokens → punctuation / section headers."""
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


# ── Inference functions ───────────────────────────────────────────────────────

def transcribe(audio):
    """MedASR: audio → mel spectrogram + CTC transcript."""
    if audio is None:
        return None, "", "", ""

    sr, raw = audio
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    audio_f32 = raw.astype(np.float32)
    if audio_f32.max() > 1.0:
        audio_f32 /= 32768.0

    if sr != SR:
        audio_f32 = librosa.resample(audio_f32, orig_sr=sr, target_sr=SR)

    duration_s = len(audio_f32) / SR
    mel = compute_mel(audio_f32)

    model_in = mel.T[np.newaxis].astype(np.float32)   # (1, T, 128)
    logits   = ASR_SESSION.run([ASR_OUTPUT], {ASR_INPUT: model_in})[0][0]  # (T', 512)

    raw_tokens, raw_text = ctc_greedy_decode(logits)
    formatted = format_transcription(raw_text)

    MAX = 80
    token_display = "  ".join(raw_tokens[:MAX])
    if len(raw_tokens) > MAX:
        token_display += f"  … (+{len(raw_tokens) - MAX} more)"

    return plot_mel(mel, duration_s), token_display, raw_text, formatted


def generate_soap(transcript: str):
    """MedGemma: transcript → SOAP note (loads model on first call)."""
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

**MedASR:** 105M Conformer CTC · INT8 ONNX · **101 MB** (↓75% from 402 MB)
**MedGemma:** 4B multimodal · GGUF Q3\_K\_M · **2.0 GB** (↓73% from 7.3 GB)

Dictate a clinical encounter → get a transcript → generate a structured SOAP note.
This is the exact pipeline running inside **[Capsule](https://github.com/mo-saif/capsule)**,
tested on a $150 Android phone (Tecno Spark 40 · 8 GB RAM · CPU only — no GPU anywhere).

> **Note:** SOAP generation loads MedGemma (2 GB) on first click — may take ~60 s on the first request.

*Part of the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/medgemma-impact-challenge) ·
Models: [MedASR](https://huggingface.co/moisf56/medasr-conformer-ctc-int8-onnx) · [MedGemma GGUF](https://huggingface.co/moisf56/medgemma-4b-q3km-gguf)*
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Capsule — MedASR + MedGemma") as demo:
    gr.Markdown(DESCRIPTION)

    # ── Step 1: Transcription ──
    gr.Markdown("### Step 1 — Transcribe")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Record or upload clinical audio",
                sources=["microphone", "upload"],
            )
            transcribe_btn = gr.Button("Transcribe with MedASR", variant="primary")

        with gr.Column(scale=2):
            mel_plot = gr.Plot(label="Log Mel Spectrogram")

    with gr.Row():
        token_output = gr.Textbox(
            label="CTC Token Stream  (raw model output — special tokens visible)",
            lines=3,
            show_copy_button=True,
        )
        raw_output = gr.Textbox(
            label="Raw CTC Text",
            lines=3,
            show_copy_button=True,
        )

    formatted_output = gr.Textbox(
        label="Formatted Transcription",
        lines=4,
        show_copy_button=True,
    )

    # ── Step 2: SOAP note ──
    gr.Markdown("### Step 2 — Generate SOAP Note with MedGemma 4B")
    soap_btn  = gr.Button("Generate SOAP Note", variant="secondary")
    soap_note = gr.Textbox(
        label="SOAP Note  (MedGemma 4B Q3_K_M)",
        lines=14,
        show_copy_button=True,
        placeholder="Click 'Generate SOAP Note' after transcribing...",
    )

    # ── Wire up ──
    transcribe_btn.click(
        fn=transcribe,
        inputs=audio_input,
        outputs=[mel_plot, token_output, raw_output, formatted_output],
    )
    soap_btn.click(
        fn=generate_soap,
        inputs=formatted_output,
        outputs=soap_note,
    )

    gr.Markdown("""
---
**Pipeline details**
| Step | Detail |
|------|--------|
| Mel spectrogram | 512-pt FFT · 128 mel bins · symmetric Hann window · log(clamp(·, 1e-5)) — matches `LasrFeatureExtractor` |
| CTC decode | Greedy argmax per frame · collapse duplicates · remove blank (token 0) · 512-token SentencePiece vocab |
| Medical formatting | `{period}` → `.` · `{comma}` → `,` · `[EXAM TYPE]` → section headers |
| SOAP generation | MedGemma 4B Q3\_K\_M · 768 max tokens · temp 0.3 · repeat penalty 1.1 |
| Quantization | MedASR: ONNX INT8 dynamic (75% smaller) · MedGemma: llama.cpp Q3\_K\_M (73% smaller) |
""")

if __name__ == "__main__":
    demo.launch()
