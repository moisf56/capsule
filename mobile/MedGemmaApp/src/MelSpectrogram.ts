/**
 * Mel spectrogram matching MedASR's LasrFeatureExtractor exactly.
 *
 * Key parameters:
 * - n_fft: 512
 * - hop_length: 160
 * - win_length: 400
 * - n_mels: 128
 * - sampling_rate: 16000
 * - Hann window: periodic=false
 * - Min clamp: 1e-5 (not 1e-10)
 * - Mel filters: from processor (pre-computed, loaded from JSON)
 */

import MEL_FILTERS from './mel_filters.json';

const N_FFT = 512;
const HOP_LENGTH = 160;
const WIN_LENGTH = 400;
const N_MELS = 128;
const N_FREQ_BINS = N_FFT / 2 + 1; // 257

// Pre-compute Hann window (periodic=false, matching PyTorch)
const hannWindow = new Float64Array(WIN_LENGTH);
for (let i = 0; i < WIN_LENGTH; i++) {
  hannWindow[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (WIN_LENGTH - 1)));
}

// Flatten mel filters from [257][128] to quick lookup
// MEL_FILTERS is number[][] shape [257, 128]
const melFiltersFlat = new Float64Array(N_FREQ_BINS * N_MELS);
for (let f = 0; f < N_FREQ_BINS; f++) {
  for (let m = 0; m < N_MELS; m++) {
    melFiltersFlat[f * N_MELS + m] = MEL_FILTERS[f][m];
  }
}

// Sparse mel filters: for each mel bin, store only non-zero freq indices + weights
// Mel filterbanks are triangular so ~95% of values are zero
const sparseMelFilters: {indices: Uint16Array; weights: Float64Array}[] = [];
for (let m = 0; m < N_MELS; m++) {
  const idx: number[] = [];
  const wts: number[] = [];
  for (let f = 0; f < N_FREQ_BINS; f++) {
    const w = MEL_FILTERS[f][m];
    if (w !== 0) {
      idx.push(f);
      wts.push(w);
    }
  }
  sparseMelFilters.push({
    indices: new Uint16Array(idx),
    weights: new Float64Array(wts),
  });
}

// Pre-compute FFT twiddle factors for each butterfly stage
const twiddleFactors: {cos: Float64Array; sin: Float64Array}[] = [];
for (let size = 2; size <= N_FFT; size *= 2) {
  const halfSize = size / 2;
  const angle = (-2 * Math.PI) / size;
  const cos = new Float64Array(halfSize);
  const sin = new Float64Array(halfSize);
  for (let j = 0; j < halfSize; j++) {
    cos[j] = Math.cos(angle * j);
    sin[j] = Math.sin(angle * j);
  }
  twiddleFactors.push({cos, sin});
}

// Pre-compute bit-reversal table for N_FFT
const bitRevTable = new Uint16Array(N_FFT);
const fftBits = Math.log2(N_FFT);
for (let i = 0; i < N_FFT; i++) {
  let result = 0;
  let x = i;
  for (let b = 0; b < fftBits; b++) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  bitRevTable[i] = result;
}

/**
 * Compute mel spectrogram matching LasrFeatureExtractor._torch_extract_fbank_features.
 *
 * Process:
 *   1. unfold(win_length, hop_length) to get frames
 *   2. Apply Hann window
 *   3. rfft(n=n_fft) per frame
 *   4. power_spec = |stft|^2
 *   5. mel_spec = clamp(power_spec @ mel_filters, min=1e-5)
 *   6. log(mel_spec)
 *
 * Optimizations:
 *   - Sparse mel filters (~10x fewer multiply-adds)
 *   - Pre-computed FFT twiddle factors + bit-reversal table
 *   - Async with periodic yields for UI updates
 */
export async function computeMelSpectrogram(
  audio: Float32Array,
  onProgress?: (percent: number) => void,
): Promise<{
  data: Float32Array;
  timeSteps: number;
  nMels: number;
}> {
  const nFrames = Math.floor((audio.length - WIN_LENGTH) / HOP_LENGTH) + 1;
  const melSpec = new Float32Array(nFrames * N_MELS);
  // Yield to UI every ~200 frames (~0.5s of work)
  const YIELD_INTERVAL = 200;

  // Reusable buffers (avoid allocation per frame)
  const paddedFrame = new Float64Array(N_FFT);
  const fftReal = new Float64Array(N_FFT);
  const fftImag = new Float64Array(N_FFT);
  // Pre-compute power spectrum buffer
  const powerSpec = new Float64Array(N_FREQ_BINS);

  for (let frame = 0; frame < nFrames; frame++) {
    const start = frame * HOP_LENGTH;

    // Windowed frame (float64 for precision, matching PyTorch)
    paddedFrame.fill(0);
    for (let i = 0; i < WIN_LENGTH; i++) {
      paddedFrame[i] = audio[start + i] * hannWindow[i];
    }

    // FFT with pre-computed twiddle factors
    fftOptimized(paddedFrame, fftReal, fftImag);

    // Power spectrum (only first N_FREQ_BINS = 257)
    for (let f = 0; f < N_FREQ_BINS; f++) {
      powerSpec[f] = fftReal[f] * fftReal[f] + fftImag[f] * fftImag[f];
    }

    // Sparse mel filter application (~10x fewer multiplications)
    for (let m = 0; m < N_MELS; m++) {
      const {indices, weights} = sparseMelFilters[m];
      let sum = 0;
      for (let k = 0; k < indices.length; k++) {
        sum += powerSpec[indices[k]] * weights[k];
      }
      melSpec[frame * N_MELS + m] = Math.log(Math.max(sum, 1e-5));
    }

    // Yield to UI thread periodically for progress updates
    if (frame % YIELD_INTERVAL === 0) {
      const pct = Math.round((frame / nFrames) * 100);
      if (onProgress) onProgress(pct);
      await new Promise(r => setTimeout(r, 0));
    }
  }

  if (onProgress) onProgress(100);
  return {data: melSpec, timeSteps: nFrames, nMels: N_MELS};
}

/**
 * Optimized Radix-2 FFT using pre-computed twiddle factors and bit-reversal table.
 * Writes results into provided output buffers (no allocation).
 */
function fftOptimized(input: Float64Array, real: Float64Array, imag: Float64Array): void {
  const n = input.length;

  // Bit-reversal permutation using pre-computed table
  imag.fill(0);
  for (let i = 0; i < n; i++) {
    real[bitRevTable[i]] = input[i];
  }

  // Butterfly operations with pre-computed twiddle factors
  let stageIdx = 0;
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const {cos, sin} = twiddleFactors[stageIdx++];

    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const evenIdx = i + j;
        const oddIdx = i + j + halfSize;

        const tr = cos[j] * real[oddIdx] - sin[j] * imag[oddIdx];
        const ti = cos[j] * imag[oddIdx] + sin[j] * real[oddIdx];

        real[oddIdx] = real[evenIdx] - tr;
        imag[oddIdx] = imag[evenIdx] - ti;
        real[evenIdx] = real[evenIdx] + tr;
        imag[evenIdx] = imag[evenIdx] + ti;
      }
    }
  }
}
