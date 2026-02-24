/**
 * CTC (Connectionist Temporal Classification) greedy decoder.
 *
 * CTC models output one probability distribution per time frame.
 * Greedy decode: argmax each frame, collapse consecutive duplicates, remove blanks.
 *
 * MedASR vocab: 512 tokens (SentencePiece), blank = token 0.
 */

// MedASR vocabulary (512 tokens from SentencePiece tokenizer)
// We load this from the tokenizer config, but for now hardcode the key tokens
// Full vocab will be loaded from tokenizer.json

/**
 * CTC greedy decode from logits.
 * @param logits - Float32Array of shape [time_steps * vocab_size]
 * @param timeSteps - number of time steps
 * @param vocabSize - vocabulary size (512 for MedASR)
 * @param vocab - vocabulary array mapping token_id -> string
 * @returns decoded text
 */
export function ctcGreedyDecode(
  logits: Float32Array,
  timeSteps: number,
  vocabSize: number,
  vocab: string[],
): string {
  const tokens: number[] = [];
  let prevId = -1;

  for (let t = 0; t < timeSteps; t++) {
    // Find argmax for this time step
    let maxVal = -Infinity;
    let maxId = 0;
    const offset = t * vocabSize;

    for (let v = 0; v < vocabSize; v++) {
      if (logits[offset + v] > maxVal) {
        maxVal = logits[offset + v];
        maxId = v;
      }
    }

    // CTC rules: skip blank (0) and skip consecutive duplicates
    if (maxId !== 0 && maxId !== prevId) {
      tokens.push(maxId);
    }
    prevId = maxId;
  }

  // Convert token IDs to text using vocab
  let text = '';
  for (const id of tokens) {
    if (id < vocab.length) {
      text += vocab[id];
    }
  }

  return text;
}

/**
 * Format MedASR raw transcription to readable text.
 * MedASR uses special tokens like {period}, {comma}, [EXAM TYPE], etc.
 */
export function formatTranscription(text: string): string {
  // Section headers: [EXAM TYPE] -> Exam Type
  text = text.replace(
    /\[([A-Z\s]+)\]/g,
    (_match, p1) =>
      p1
        .toLowerCase()
        .replace(/\b\w/g, (c: string) => c.toUpperCase()),
  );

  // Strip special tokens
  text = text.replace(/<\/s>/g, '');
  text = text.replace(/<s>/g, '');
  text = text.replace(/<epsilon>/g, '');
  text = text.replace(/<pad>/g, '');

  // Formatting tokens
  text = text.replace(/\{period\}/g, '.');
  text = text.replace(/\{comma\}/g, ',');
  text = text.replace(/\{colon\}/g, ':');
  text = text.replace(/\{semicolon\}/g, ';');
  text = text.replace(/\{new paragraph\}/g, '\n\n');
  text = text.replace(/\{question mark\}/g, '?');
  text = text.replace(/\{exclamation point\}/g, '!');
  text = text.replace(/\{hyphen\}/g, '-');

  // Clean remaining braces
  text = text.replace(/\{[^}]*\}/g, '');

  // Clean whitespace
  text = text.replace(/\s+/g, ' ').trim();
  text = text.replace(/\s+([.,:;?!-])/g, '$1');

  return text;
}
