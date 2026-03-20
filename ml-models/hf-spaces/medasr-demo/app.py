"""
Capsule — MedASR Demo
Gradio Space: moisf56/capsule-medasr-demo

Pipeline:
  Audio → LasrFeatureExtractor → google/medasr (Conformer CTC)
        → Manual CTC greedy decode → Medical Text Formatting
"""

import re

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.signal.windows
import soundfile as sf
import torch
import transformers

matplotlib.use("Agg")

# ── MedASR ────────────────────────────────────────────────────────────────────

MODEL_ID = "google/medasr"
print(f"Loading MedASR ({MODEL_ID})...")
_fe    = transformers.LasrFeatureExtractor.from_pretrained(MODEL_ID)
_model = transformers.AutoModelForCTC.from_pretrained(MODEL_ID)
_tok   = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
_model.eval()
print("MedASR ready.")

# ── Audio parameters ──────────────────────────────────────────────────────────

SR         = 16_000
N_FFT      = 512
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS     = 128

_HANN = scipy.signal.windows.hann(WIN_LENGTH, sym=True).astype(np.float64)


def _make_mel_filters(sr: int, n_fft: int, n_mels: int,
                      fmin: float = 0.0, fmax: float = None) -> np.ndarray:
    """Pure-numpy mel filterbank (Slaney normalization, matches librosa default)."""
    if fmax is None:
        fmax = sr / 2.0
    def hz2mel(f): return 2595.0 * np.log10(1.0 + f / 700.0)
    def mel2hz(m): return 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    freqs   = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_f   = hz2mel(freqs)
    mel_pts = np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2)
    W = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, mid, hi = mel_pts[i], mel_pts[i + 1], mel_pts[i + 2]
        W[i] = np.maximum(0, np.minimum(
            (mel_f - lo) / (mid - lo),
            (hi - mel_f) / (hi - mid),
        ))
    # Slaney normalization
    enorm = 2.0 / (mel2hz(mel_pts[2:]) - mel2hz(mel_pts[:-2]))
    W *= enorm[:, None]
    return W.astype(np.float64)


_MEL_FILTERS = _make_mel_filters(SR, N_FFT, N_MELS)


# ── DSP ───────────────────────────────────────────────────────────────────────

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


def compute_rms(audio: np.ndarray, hop: int = HOP_LENGTH) -> np.ndarray:
    n_frames = len(audio) // hop
    rms = np.array([
        np.sqrt(np.mean(audio[i * hop : (i + 1) * hop] ** 2))
        for i in range(n_frames)
    ])
    return rms


# ── CTC decode ────────────────────────────────────────────────────────────────

def ctc_greedy_decode(logits_tensor: torch.Tensor) -> str:
    ids = logits_tensor.argmax(dim=-1).tolist()
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


# ── Visualisations ────────────────────────────────────────────────────────────

BG    = "#0f172a"
GRID  = "#1e293b"
WHITE = "white"
DIM   = "#94a3b8"


def plot_mel(mel: np.ndarray, duration_s: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 3))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    img = ax.imshow(
        mel, aspect="auto", origin="lower", cmap="magma",
        extent=[0, duration_s, 0, SR / 2 / 1000],
        interpolation="nearest",
    )
    ax.set_xlabel("Time (s)", color=WHITE, fontsize=10)
    ax.set_ylabel("Frequency (kHz)", color=WHITE, fontsize=10)
    ax.set_title(
        f"Log Mel Spectrogram  ·  {mel.shape[1]} frames  ·  {N_MELS} mel bins  ·  {N_FFT}-pt FFT",
        color=WHITE, fontsize=11, pad=8,
    )
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Log Energy", color=WHITE, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
    plt.tight_layout()
    return fig


def plot_waveform_rms(audio: np.ndarray, duration_s: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 2.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    t_wave = np.linspace(0, duration_s, len(audio))
    ax.plot(t_wave, audio, color="#38bdf8", linewidth=0.4, alpha=0.6, label="Waveform")

    rms  = compute_rms(audio)
    t_rms = np.linspace(0, duration_s, len(rms))
    ax.plot(t_rms,  rms, color="#f97316", linewidth=1.8, label="RMS energy")
    ax.plot(t_rms, -rms, color="#f97316", linewidth=1.8)
    ax.fill_between(t_rms,  rms, -rms, color="#f97316", alpha=0.15)

    # speech activity threshold line
    thresh = 0.02 * rms.max()
    ax.axhline( thresh, color="#4ade80", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.axhline(-thresh, color="#4ade80", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_xlim(0, duration_s)
    ax.set_xlabel("Time (s)", color=WHITE, fontsize=10)
    ax.set_ylabel("Amplitude", color=WHITE, fontsize=10)
    ax.set_title("Waveform  +  RMS Energy Envelope", color=WHITE, fontsize=11, pad=8)
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.legend(loc="upper right", fontsize=8, facecolor=GRID, labelcolor=WHITE,
              framealpha=0.7)
    plt.tight_layout()
    return fig


def plot_ctc_heatmap(logits_tensor: torch.Tensor, duration_s: float) -> plt.Figure:
    """Heatmap of CTC probabilities: time × top-N vocab tokens."""
    probs = torch.softmax(logits_tensor, dim=-1).numpy()   # (T, vocab)
    T     = probs.shape[0]

    # Pick top 20 tokens by mean activation (excluding blank=0)
    mean_prob   = probs[:, 1:].mean(axis=0)
    top_ids_rel = np.argsort(mean_prob)[-20:][::-1]
    top_ids     = top_ids_rel + 1                          # shift back (skip blank)

    labels = []
    for id_ in top_ids:
        tok = _tok.convert_ids_to_tokens([int(id_)])[0]
        tok = tok.replace("▁", "·").replace("{", "").replace("}", "")
        labels.append(tok[:10])

    heat = probs[:, top_ids].T   # (20, T)

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    t_axis = np.linspace(0, duration_s, T)
    img = ax.imshow(
        heat, aspect="auto", origin="upper", cmap="viridis",
        extent=[0, duration_s, len(top_ids) - 0.5, -0.5],
        interpolation="nearest",
        vmin=0, vmax=heat.max(),
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8, color=WHITE)
    ax.set_xlabel("Time (s)", color=WHITE, fontsize=10)
    ax.set_title(
        f"CTC Output Probabilities  ·  top 20 tokens  ·  {T} output frames",
        color=WHITE, fontsize=11, pad=8,
    )
    ax.tick_params(colors=WHITE)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Probability", color=WHITE, fontsize=9)
    cbar.ax.yaxis.set_tick_params(color=WHITE)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=WHITE)
    plt.tight_layout()
    return fig


def build_quality_card(audio: np.ndarray, duration_s: float) -> str:
    rms       = compute_rms(audio)
    thresh    = 0.02 * rms.max()
    speech_ratio = float((rms > thresh).mean())

    # SNR: ratio of speech-frame RMS to noise-floor RMS (in dB)
    speech_rms = rms[rms > thresh]
    noise_rms  = rms[rms <= thresh]
    if len(speech_rms) > 0 and len(noise_rms) > 0 and noise_rms.mean() > 0:
        snr_db = 20 * np.log10(speech_rms.mean() / noise_rms.mean())
    else:
        snr_db = float("nan")

    peak_db = 20 * np.log10(np.abs(audio).max() + 1e-9)

    def badge(val, good, warn):
        if val >= good:  color = "#4ade80"
        elif val >= warn: color = "#fbbf24"
        else:             color = "#f87171"
        return color

    sr_color  = badge(speech_ratio, 0.5, 0.25)
    snr_color = badge(snr_db, 20, 10) if not np.isnan(snr_db) else "#94a3b8"

    snr_str = f"{snr_db:.1f} dB" if not np.isnan(snr_db) else "—"

    return f"""
<div style="display:flex;gap:12px;flex-wrap:wrap;font-family:monospace;margin:4px 0">
  <div style="background:#1e293b;border-radius:8px;padding:10px 16px;min-width:110px">
    <div style="color:#94a3b8;font-size:11px">DURATION</div>
    <div style="color:white;font-size:18px;font-weight:600">{duration_s:.1f} s</div>
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:10px 16px;min-width:110px">
    <div style="color:#94a3b8;font-size:11px">SAMPLE RATE</div>
    <div style="color:white;font-size:18px;font-weight:600">{SR//1000} kHz</div>
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:10px 16px;min-width:110px">
    <div style="color:#94a3b8;font-size:11px">SPEECH RATIO</div>
    <div style="color:{sr_color};font-size:18px;font-weight:600">{speech_ratio:.0%}</div>
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:10px 16px;min-width:110px">
    <div style="color:#94a3b8;font-size:11px">EST. SNR</div>
    <div style="color:{snr_color};font-size:18px;font-weight:600">{snr_str}</div>
  </div>
  <div style="background:#1e293b;border-radius:8px;padding:10px 16px;min-width:110px">
    <div style="color:#94a3b8;font-size:11px">PEAK LEVEL</div>
    <div style="color:white;font-size:18px;font-weight:600">{peak_db:.1f} dB</div>
  </div>
</div>"""


# ── Inference ─────────────────────────────────────────────────────────────────

def transcribe(audio_path):
    if audio_path is None:
        return None, None, None, "", "", ""

    audio_raw, native_sr = sf.read(audio_path, always_2d=True, dtype="float32")
    audio_f32 = audio_raw.mean(axis=1)          # mono
    if native_sr != SR:                          # resample to 16 kHz
        n_out     = int(len(audio_f32) * SR / native_sr)
        audio_f32 = scipy.signal.resample(audio_f32, n_out)
    audio_f32 = audio_f32.astype(np.float32)
    duration_s   = len(audio_f32) / SR
    mel          = compute_mel(audio_f32)

    inputs  = _fe(audio_f32, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        logits = _model(**inputs).logits[0]   # (T, vocab)

    raw_text  = ctc_greedy_decode(logits)
    formatted = format_transcription(raw_text)

    print(f"[ASR] raw:       {raw_text[:200]}")
    print(f"[ASR] formatted: {formatted[:200]}")

    fig_mel  = plot_mel(mel, duration_s)
    fig_wave = plot_waveform_rms(audio_f32, duration_s)
    fig_ctc  = plot_ctc_heatmap(logits, duration_s)
    quality  = build_quality_card(audio_f32, duration_s)

    return fig_wave, fig_mel, fig_ctc, quality, formatted, raw_text


# ── Gradio UI ─────────────────────────────────────────────────────────────────

DESCRIPTION = """
## 🎙️ Capsule — Medical Speech Recognition *(ASR component)*

**[Capsule](https://github.com/moisf56/capsule)** is a fully on-device, multimodal clinical documentation tool —
it listens to a doctor-patient encounter, transcribes it with MedASR, then generates a structured SOAP note using MedGemma.
Everything runs locally on a **$150 Android phone** (Tecno Spark 40 · 8 GB RAM · CPU only — no cloud, no GPU).

**This demo shows the ASR component only.**

| Component | Model | Format | Size |
|-----------|-------|--------|------|
| 🎙️ Speech → Text | `google/medasr` · 105M Conformer CTC | [ONNX INT8](https://huggingface.co/moisf56/medasr-conformer-ctc-int8-onnx) | 101 MB ↓75% |
| 🧠 Text → SOAP note | `google/medgemma-1.5-4b-it` | [GGUF Q3_K_M](https://huggingface.co/moisf56/medgemma-4b-q3km-gguf) | 2.0 GB ↓73% |

*Part of the [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)*
"""

with gr.Blocks(title="Capsule — MedASR") as demo:
    gr.Markdown(DESCRIPTION)

    # ── Input ──
    gr.Markdown("### Step 1 — Record or upload audio")
    with gr.Row():
        with gr.Column(scale=1):
            audio_input    = gr.Audio(
                label="Clinical audio (any sample rate · mono or stereo)",
                sources=["microphone", "upload"],
                type="filepath",
            )
            transcribe_btn = gr.Button("▶  Transcribe with MedASR", variant="primary")

    # ── Signal analysis ──
    gr.Markdown("### Signal Analysis")
    quality_card = gr.HTML()
    with gr.Row():
        fig_wave = gr.Plot(label="Waveform + RMS Energy Envelope")
    with gr.Row():
        fig_mel  = gr.Plot(label="Log Mel Spectrogram")
    with gr.Row():
        fig_ctc  = gr.Plot(label="CTC Output Probabilities (top 20 tokens)")

    # ── Transcript ──
    gr.Markdown("### Step 2 — Review and edit transcript")
    gr.Markdown("_MedASR output below. Edit before use in clinical documentation._")
    formatted_output = gr.Textbox(
        label="Transcript  ✏️ editable",
        lines=6,
        interactive=True,
        placeholder="Transcript will appear here after Step 1...",
    )
    with gr.Accordion("Raw ASR output (before formatting)", open=False):
        raw_output = gr.Textbox(
            label="Raw MedASR text  (special tokens visible)",
            lines=3,
        )

    transcribe_btn.click(
        fn=transcribe,
        inputs=[audio_input],
        outputs=[fig_wave, fig_mel, fig_ctc, quality_card, formatted_output, raw_output],
        api_name=False,
    )

    gr.Markdown("""
---
**Pipeline details**
| Step | Detail |
|------|--------|
| Feature extraction | `LasrFeatureExtractor` · 512-pt FFT · 128 mel bins · symmetric Hann window · log(clamp(·, 1e-5)) |
| Model | `google/medasr` · 105M Conformer CTC · trained on medical speech |
| CTC decode | Greedy argmax per frame · collapse consecutive duplicates · remove blank (token 0) |
| Medical NLP | `{period}` → `.` · `{comma}` → `,` · `[EXAM TYPE]` → section headers |
| On-device mobile | MedASR ONNX INT8 (101 MB, ↓75%) → transcript · MedGemma GGUF Q3_K_M (2.0 GB, ↓73%) → SOAP note *(full pipeline in the app)* |
""")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
