"""
MedASR - Medical Speech-to-Text

105M parameter Conformer ASR model for medical dictation.
Uses beam search with KenLM language model for best quality (~5% WER).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import dataclasses
import re
import librosa
import torch
import huggingface_hub
import pyctcdecode
import transformers


def _restore_text(text: str) -> str:
    """Convert tokenizer output back to readable text."""
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


class LasrCtcBeamSearchDecoder:
    """CTC beam search decoder with KenLM language model."""
    def __init__(self, tokenizer, kenlm_model_path=None, **kwargs):
        vocab = [None for _ in range(tokenizer.vocab_size)]
        for k, v in tokenizer.vocab.items():
            if v < tokenizer.vocab_size:
                vocab[v] = k
        assert not [i for i in vocab if i is None]
        vocab[0] = ""
        for i in range(1, len(vocab)):
            piece = vocab[i]
            if not piece.startswith("<") and not piece.endswith(">"):
                piece = "▁" + piece.replace("▁", "#")
            vocab[i] = piece
        self._decoder = pyctcdecode.build_ctcdecoder(vocab, kenlm_model_path, **kwargs)

    def decode_beams(self, *args, **kwargs):
        beams = self._decoder.decode_beams(*args, **kwargs)
        return [dataclasses.replace(i, text=_restore_text(i.text)) for i in beams]


def create_medasr_pipeline(model_id="google/medasr"):
    """Create MedASR pipeline with beam search + language model."""
    lm_path = huggingface_hub.hf_hub_download(model_id, 'lm_6.kenlm')
    feature_extractor = transformers.LasrFeatureExtractor.from_pretrained(model_id)
    feature_extractor._processor_class = "LasrProcessorWithLM"
    pipe = transformers.pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        feature_extractor=feature_extractor,
        decoder=LasrCtcBeamSearchDecoder(
            transformers.AutoTokenizer.from_pretrained(model_id), lm_path
        ),
    )
    return pipe


def format_transcription(text: str) -> str:
    """Convert MedASR formatting tokens to punctuation."""
    # Section headers: [EXAM TYPE] -> Exam type
    text = re.sub(r'\[([A-Z\s]+)\]', lambda m: m.group(1).title(), text)

    # Formatting tokens
    text = text.replace("{period}", ".")
    text = text.replace("{comma}", ",")
    text = text.replace("{colon}", ":")
    text = text.replace("{new paragraph}", "\n\n")

    # Clean up any remaining braces
    text = re.sub(r'\{[^}]*\}', '', text)

    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([.,:])', r'\1', text)

    return text


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()

    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i-1] == hyp[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + 1)

    return d[len(ref)][len(hyp)] / len(ref) if ref else 0


def transcribe(audio_path: str, pipe=None, beam_width: int = 8) -> str:
    """
    Transcribe audio file to text.

    Args:
        audio_path: Path to audio file (wav, mp3, etc.)
        pipe: MedASR pipeline (created if not provided)
        beam_width: Beam search width (default 8)

    Returns:
        Formatted transcription text
    """
    if pipe is None:
        pipe = create_medasr_pipeline()

    audio, sr = librosa.load(audio_path, sr=16000)

    result = pipe(
        {"raw": audio, "sampling_rate": sr},
        chunk_length_s=20,
        stride_length_s=2,
        decoder_kwargs=dict(beam_width=beam_width),
    )

    return format_transcription(result["text"])


def main():
    """Test MedASR with sample audio."""
    print("MedASR - Medical Speech Recognition")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Reference transcript
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

    # Download test audio
    print("Loading model and audio...")
    audio_path = huggingface_hub.hf_hub_download("google/medasr", 'test_audio.wav')

    # Create pipeline
    pipe = create_medasr_pipeline()

    # Transcribe
    print("Transcribing...")
    transcription = transcribe(audio_path, pipe)

    print("\n" + "=" * 60)
    print("TRANSCRIPTION:")
    print("=" * 60)
    print(transcription)
    print("=" * 60)

    # Calculate WER
    wer = calculate_wer(reference, transcription)
    print(f"\nWord Error Rate: {wer:.1%}")

    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")


if __name__ == "__main__":
    main()
