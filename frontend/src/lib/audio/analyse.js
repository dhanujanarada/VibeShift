// src/lib/audio/analyse.js
// Pure Web Audio API + pure-JS FFT audio feature extraction.
// No external dependencies.

const FRAME_SIZE = 2048;
const HOP_SIZE = 512;
const ROLLOFF_PERCENT = 0.85;

// ---------------------------------------------------------------------------
// Radix-2 Cooley-Tukey FFT (in-place, power-of-2 length)
// Returns { re: Float32Array, im: Float32Array }
// ---------------------------------------------------------------------------
function fft(signal) {
  const N = signal.length;
  const re = new Float32Array(signal);
  const im = new Float32Array(N);

  // Bit-reversal permutation
  let j = 0;
  for (let i = 1; i < N; i++) {
    let bit = N >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let tmp = re[i]; re[i] = re[j]; re[j] = tmp;
      tmp = im[i]; im[i] = im[j]; im[j] = tmp;
    }
  }

  // Cooley-Tukey butterfly stages
  for (let len = 2; len <= N; len <<= 1) {
    const halfLen = len >> 1;
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < N; i += len) {
      let curRe = 1.0;
      let curIm = 0.0;
      for (let k = 0; k < halfLen; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + halfLen] * curRe - im[i + k + halfLen] * curIm;
        const vIm = re[i + k + halfLen] * curIm + im[i + k + halfLen] * curRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + halfLen] = uRe - vRe;
        im[i + k + halfLen] = uIm - vIm;
        const nextRe = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nextRe;
      }
    }
  }

  return { re, im };
}

// ---------------------------------------------------------------------------
// Hann window coefficients
// ---------------------------------------------------------------------------
function hannWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  }
  return w;
}

// ---------------------------------------------------------------------------
// Main analysis function
// ---------------------------------------------------------------------------
export async function analyseAudio(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch audio: ${response.status}`);
  const arrayBuffer = await response.arrayBuffer();

  const ctx = new AudioContext();
  const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
  await ctx.close();

  const sampleRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;

  // Mix down to mono
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  for (let ch = 0; ch < numChannels; ch++) {
    const chData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += chData[i];
    }
  }
  if (numChannels > 1) {
    for (let i = 0; i < length; i++) mono[i] /= numChannels;
  }

  const hann = hannWindow(FRAME_SIZE);
  const numFrames = Math.max(0, Math.floor((length - FRAME_SIZE) / HOP_SIZE) + 1);

  const rms              = new Float32Array(numFrames);
  const spectralCentroid = new Float32Array(numFrames);
  const spectralRolloff  = new Float32Array(numFrames);
  const spectralFlatness = new Float32Array(numFrames);
  const zcr              = new Float32Array(numFrames);

  const halfN = FRAME_SIZE / 2;
  const frame = new Float32Array(FRAME_SIZE);

  for (let f = 0; f < numFrames; f++) {
    const start = f * HOP_SIZE;

    // Apply Hann window and compute RMS simultaneously
    let sumSq = 0;
    let crossings = 0;
    for (let i = 0; i < FRAME_SIZE; i++) {
      const s = mono[start + i] ?? 0;
      frame[i] = s * hann[i];
      sumSq += s * s;
    }

    // RMS (on raw, not windowed, samples)
    rms[f] = Math.sqrt(sumSq / FRAME_SIZE);

    // Zero crossing rate (on raw samples)
    for (let i = 1; i < FRAME_SIZE; i++) {
      const prev = mono[start + i - 1] ?? 0;
      const curr = mono[start + i] ?? 0;
      if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) crossings++;
    }
    zcr[f] = crossings / FRAME_SIZE;

    // FFT
    const { re, im } = fft(frame);

    // Magnitude spectrum (positive frequencies only: bins 0..halfN)
    let totalMag = 0;
    let totalPower = 0;
    let weightedBin = 0;
    let logSumPower = 0;
    const mags = new Float32Array(halfN);
    const powers = new Float32Array(halfN);

    for (let k = 0; k < halfN; k++) {
      const mag = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
      const power = mag * mag;
      mags[k] = mag;
      powers[k] = power;
      totalMag += mag;
      totalPower += power;
      weightedBin += k * mag;
      // For geometric mean: sum of log(power + epsilon)
      logSumPower += Math.log(power + 1e-10);
    }

    // Spectral Centroid (Hz)
    if (totalMag > 0) {
      const centroidBin = weightedBin / totalMag;
      spectralCentroid[f] = (centroidBin * sampleRate) / FRAME_SIZE;
    } else {
      spectralCentroid[f] = 0;
    }

    // Spectral Rolloff (Hz)
    const rolloffThreshold = ROLLOFF_PERCENT * totalPower;
    let cumPower = 0;
    let rolloffBin = 0;
    for (let k = 0; k < halfN; k++) {
      cumPower += powers[k];
      if (cumPower >= rolloffThreshold) {
        rolloffBin = k;
        break;
      }
    }
    spectralRolloff[f] = (rolloffBin * sampleRate) / FRAME_SIZE;

    // Spectral Flatness (Wiener entropy)
    if (totalPower > 0) {
      const arith = totalPower / halfN;
      const geomLog = logSumPower / halfN;
      const geom = Math.exp(geomLog);
      const flatness = arith > 0 ? geom / arith : 0;
      spectralFlatness[f] = Math.min(1, Math.max(0, flatness));
    } else {
      spectralFlatness[f] = 0;
    }
  }

  // Scalar means
  const mean = (arr) => {
    if (arr.length === 0) return 0;
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    return s / arr.length;
  };

  return {
    rms,
    spectralCentroid,
    spectralRolloff,
    spectralFlatness,
    zcr,
    meanRms:              mean(rms),
    meanSpectralCentroid: mean(spectralCentroid),
    meanSpectralRolloff:  mean(spectralRolloff),
    meanSpectralFlatness: mean(spectralFlatness),
    meanZcr:              mean(zcr),
    durationSec:  length / sampleRate,
    sampleRate,
    numFrames,
  };
}
