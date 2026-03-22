<script>
  // MelSpectrogram.svelte
  // Pure Web Audio API + Canvas spectrogram — no external libs.

  let { src, label } = $props();

  let canvas = $state(null);
  let loadState = $state("idle"); // idle | computing | done | error
  let errorText = $state("");

  // ── Cooley-Tukey radix-2 FFT (in-place, complex input) ───────────────────
  // re[] and im[] are real/imaginary parts, length must be a power of 2.
  function fft(re, im) {
    const n = re.length;
    // Bit-reversal permutation
    let j = 0;
    for (let i = 1; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    // Butterfly passes
    for (let len = 2; len <= n; len <<= 1) {
      const half = len >> 1;
      const ang = (-2 * Math.PI) / len;
      const wRe = Math.cos(ang);
      const wIm = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let curRe = 1, curIm = 0;
        for (let k = 0; k < half; k++) {
          const uRe = re[i + k];
          const uIm = im[i + k];
          const vRe = re[i + k + half] * curRe - im[i + k + half] * curIm;
          const vIm = re[i + k + half] * curIm + im[i + k + half] * curRe;
          re[i + k]        = uRe + vRe;
          im[i + k]        = uIm + vIm;
          re[i + k + half] = uRe - vRe;
          im[i + k + half] = uIm - vIm;
          const nextRe = curRe * wRe - curIm * wIm;
          curIm = curRe * wIm + curIm * wRe;
          curRe = nextRe;
        }
      }
    }
  }

  // ── Hann window ───────────────────────────────────────────────────────────
  function hannWindow(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
    return w;
  }

  // ── Build mel filterbank matrix (nMel × (fftSize/2+1)) ───────────────────
  function melFilterbank(nMel, fftSize, sampleRate) {
    const nFft = fftSize / 2 + 1;
    const fMin = 0;
    const fMax = sampleRate / 2;

    const hzToMel = (f) => 2595 * Math.log10(1 + f / 700);
    const melToHz = (m) => 700 * (10 ** (m / 2595) - 1);

    const melMin = hzToMel(fMin);
    const melMax = hzToMel(fMax);

    // nMel+2 equally-spaced mel points → Hz
    const melPts = new Float64Array(nMel + 2);
    for (let i = 0; i <= nMel + 1; i++) {
      melPts[i] = melToHz(melMin + (i / (nMel + 1)) * (melMax - melMin));
    }

    // Map mel Hz points to FFT bin indices
    const bin = melPts.map((f) => Math.floor((f / (sampleRate / 2)) * (nFft - 1)));

    // Build nMel triangular filters
    const fb = [];
    for (let m = 1; m <= nMel; m++) {
      const filter = new Float32Array(nFft);
      for (let k = bin[m - 1]; k < bin[m]; k++) {
        filter[k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1] + 1e-9);
      }
      for (let k = bin[m]; k <= bin[m + 1]; k++) {
        filter[k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m] + 1e-9);
      }
      fb.push(filter);
    }
    return fb;
  }

  // ── Viridis-like colormap (black→purple→orange→yellow) ───────────────────
  function colormap(t) {
    // Piecewise linear approximation of inferno
    const stops = [
      [0,     0,   0,   4],
      [0.25,  71,  14, 134],
      [0.5,  181,  55,  58],
      [0.75, 245, 135,  26],
      [1.0,  252, 255, 164],
    ];
    t = Math.max(0, Math.min(1, t));
    let i = 0;
    while (i < stops.length - 2 && t > stops[i + 1][0]) i++;
    const [t0, r0, g0, b0] = stops[i];
    const [t1, r1, g1, b1] = stops[i + 1];
    const f = (t - t0) / (t1 - t0);
    return [
      Math.round(r0 + f * (r1 - r0)),
      Math.round(g0 + f * (g1 - g0)),
      Math.round(b0 + f * (b1 - b0)),
    ];
  }

  // ── Main computation ──────────────────────────────────────────────────────
  async function compute(url) {
    if (!url || !canvas) return;

    loadState = "computing";
    errorText = "";

    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const arrayBuf = await response.arrayBuffer();

      const audioCtx = new AudioContext();
      const decoded  = await audioCtx.decodeAudioData(arrayBuf);
      audioCtx.close();

      // Always use mono channel 0
      const samples    = decoded.getChannelData(0);
      const sampleRate = decoded.sampleRate;

      const FFT_SIZE = 2048;
      const HOP_SIZE = 512;
      const N_MEL    = 128;
      const nFft     = FFT_SIZE / 2 + 1;

      const hann = hannWindow(FFT_SIZE);
      const fb   = melFilterbank(N_MEL, FFT_SIZE, sampleRate);

      // How many frames fit?
      const nFrames = Math.max(1, Math.floor((samples.length - FFT_SIZE) / HOP_SIZE) + 1);

      // Spectrogram: nFrames × N_MEL
      const spec = new Float32Array(nFrames * N_MEL);

      const re = new Float64Array(FFT_SIZE);
      const im = new Float64Array(FFT_SIZE);

      for (let f = 0; f < nFrames; f++) {
        const start = f * HOP_SIZE;
        // Fill windowed frame
        for (let i = 0; i < FFT_SIZE; i++) {
          re[i] = (start + i < samples.length ? samples[start + i] : 0) * hann[i];
          im[i] = 0;
        }

        fft(re, im);

        // Power spectrum (only positive frequencies)
        const power = new Float32Array(nFft);
        for (let k = 0; k < nFft; k++) {
          power[k] = re[k] * re[k] + im[k] * im[k];
        }

        // Apply mel filterbank, convert to dB
        for (let m = 0; m < N_MEL; m++) {
          let energy = 0;
          const filter = fb[m];
          for (let k = 0; k < nFft; k++) energy += filter[k] * power[k];
          spec[f * N_MEL + m] = 10 * Math.log10(energy + 1e-9);
        }
      }

      // Normalise to [0, 1]
      let sMin = Infinity, sMax = -Infinity;
      for (let i = 0; i < spec.length; i++) {
        if (spec[i] < sMin) sMin = spec[i];
        if (spec[i] > sMax) sMax = spec[i];
      }
      const range = sMax - sMin || 1;

      // Draw onto canvas
      const W = canvas.clientWidth || 400;
      const H = 160;
      canvas.width  = W;
      canvas.height = H;

      const ctx = canvas.getContext("2d");
      const img = ctx.createImageData(W, H);

      for (let px = 0; px < W; px++) {
        const frame = Math.min(nFrames - 1, Math.floor((px / W) * nFrames));
        for (let py = 0; py < H; py++) {
          // py=0 is top → high freq; py=H-1 is bottom → low freq
          const melBin = Math.floor(((H - 1 - py) / H) * N_MEL);
          const val    = (spec[frame * N_MEL + melBin] - sMin) / range;
          const [r, g, b] = colormap(val);
          const idx = (py * W + px) * 4;
          img.data[idx]     = r;
          img.data[idx + 1] = g;
          img.data[idx + 2] = b;
          img.data[idx + 3] = 255;
        }
      }

      ctx.putImageData(img, 0, 0);
      loadState = "done";
    } catch (e) {
      errorText = e.message || "Failed to decode audio.";
      loadState = "error";
    }
  }

  $effect(() => {
    compute(src);
  });
</script>

<div class="mel-container">
  <span class="mel-label">{label}</span>

  {#if loadState === "computing"}
    <div class="mel-status">Computing…</div>
  {:else if loadState === "error"}
    <div class="mel-status error">{errorText}</div>
  {/if}

  <canvas
    bind:this={canvas}
    style:display={loadState === "done" ? "block" : "none"}
  ></canvas>
</div>

<style>
  .mel-container {
    background: #1e1e2e;
    border-radius: 8px;
    padding: 0.75rem;
  }

  .mel-label {
    display: block;
    font-size: 0.7rem;
    font-variant: small-caps;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: #cba6f7;
    margin-bottom: 0.5rem;
  }

  canvas {
    width: 100%;
    height: 160px;
    display: block;
  }

  .mel-status {
    height: 160px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #585b70;
    font-size: 0.85rem;
  }

  .mel-status.error {
    color: #f38ba8;
  }
</style>
