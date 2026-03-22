<!-- src/lib/components/AudioAnalysis.svelte -->
<script>
  import { analyseAudio } from '$lib/audio/analyse.js';

  let { inputSrc, outputSrc } = $props();

  let inputMetrics  = $state(null);
  let outputMetrics = $state(null);
  let loading       = $state(false);
  let error         = $state(null);

  // Canvas refs
  let rmsCanvas       = $state(null);
  let centroidCanvas  = $state(null);
  let barCanvas       = $state(null);

  const INPUT_COLOR  = '#89b4fa';
  const OUTPUT_COLOR = '#a6e3a1';
  const BG_COLOR     = '#1e1e2e';
  const AXIS_COLOR   = '#45475a';
  const LABEL_COLOR  = '#a6adc8';
  const CANVAS_H     = 180;

  // -------------------------------------------------------------------------
  // Fetch + analyse both tracks whenever the src props change
  // -------------------------------------------------------------------------
  $effect(() => {
    const src1 = inputSrc;
    const src2 = outputSrc;
    if (!src1 || !src2) return;

    loading = true;
    error   = null;
    inputMetrics  = null;
    outputMetrics = null;

    Promise.all([analyseAudio(src1), analyseAudio(src2)])
      .then(([im, om]) => {
        inputMetrics  = im;
        outputMetrics = om;
        loading       = false;
      })
      .catch((e) => {
        error   = e.message ?? 'Analysis failed';
        loading = false;
      });
  });

  // -------------------------------------------------------------------------
  // Drawing helpers
  // -------------------------------------------------------------------------

  /** Draw a two-line time-series chart on a canvas element. */
  function drawLineChart(canvas, metrics1, metrics2, accessor, yLabel) {
    if (!canvas) return;
    const W = canvas.clientWidth || canvas.width;
    canvas.width  = W;
    canvas.height = CANVAS_H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, CANVAS_H);
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, W, CANVAS_H);

    const PAD_L = 48, PAD_R = 16, PAD_T = 16, PAD_B = 32;
    const plotW = W - PAD_L - PAD_R;
    const plotH = CANVAS_H - PAD_T - PAD_B;

    const arr1 = accessor(metrics1);
    const arr2 = accessor(metrics2);

    // Global min/max
    let yMin = Infinity, yMax = -Infinity;
    for (let i = 0; i < arr1.length; i++) { yMin = Math.min(yMin, arr1[i]); yMax = Math.max(yMax, arr1[i]); }
    for (let i = 0; i < arr2.length; i++) { yMin = Math.min(yMin, arr2[i]); yMax = Math.max(yMax, arr2[i]); }
    if (yMax === yMin) yMax = yMin + 1;

    const dur1 = metrics1.durationSec;
    const dur2 = metrics2.durationSec;
    const totalDur = Math.max(dur1, dur2);

    const toX = (t) => PAD_L + (t / totalDur) * plotW;
    const toY = (v) => PAD_T + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

    // Axes
    ctx.strokeStyle = AXIS_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD_L, PAD_T);
    ctx.lineTo(PAD_L, PAD_T + plotH);
    ctx.lineTo(PAD_L + plotW, PAD_T + plotH);
    ctx.stroke();

    // Y ticks (4)
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'right';
    for (let t = 0; t <= 4; t++) {
      const v = yMin + (t / 4) * (yMax - yMin);
      const y = toY(v);
      ctx.fillStyle = AXIS_COLOR;
      ctx.beginPath(); ctx.moveTo(PAD_L - 3, y); ctx.lineTo(PAD_L, y); ctx.stroke();
      ctx.fillStyle = LABEL_COLOR;
      ctx.fillText(formatAxisVal(v), PAD_L - 5, y + 3);
    }

    // X ticks (4)
    ctx.textAlign = 'center';
    for (let t = 0; t <= 4; t++) {
      const sec = (t / 4) * totalDur;
      const x = toX(sec);
      ctx.fillStyle = AXIS_COLOR;
      ctx.beginPath(); ctx.moveTo(x, PAD_T + plotH); ctx.lineTo(x, PAD_T + plotH + 3); ctx.stroke();
      ctx.fillStyle = LABEL_COLOR;
      ctx.fillText(sec.toFixed(1) + 's', x, PAD_T + plotH + 13);
    }

    // Y axis label
    ctx.save();
    ctx.translate(10, PAD_T + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = LABEL_COLOR;
    ctx.font = '9px Inter, sans-serif';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Lines
    function drawLine(arr, dur, color) {
      const n = arr.length;
      if (n === 0) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const t = (i / (n - 1 || 1)) * dur;
        const x = toX(t);
        const y = toY(arr[i]);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    drawLine(arr1, dur1, INPUT_COLOR);
    drawLine(arr2, dur2, OUTPUT_COLOR);

    // Legend top-right
    const legendX = PAD_L + plotW - 4;
    const legendY = PAD_T + 8;
    const swatchW = 14, swatchH = 3, gap = 4;
    [
      { color: INPUT_COLOR,  label: 'Input' },
      { color: OUTPUT_COLOR, label: 'Output' },
    ].forEach(({ color, label }, idx) => {
      const lx = legendX - (idx === 0 ? 100 : 48);
      ctx.fillStyle = color;
      ctx.fillRect(lx, legendY - 1, swatchW, swatchH);
      ctx.fillStyle = LABEL_COLOR;
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillText(label, lx + swatchW + gap, legendY + 3);
    });
  }

  /** Format axis tick values compactly. */
  function formatAxisVal(v) {
    if (Math.abs(v) >= 10000) return (v / 1000).toFixed(1) + 'k';
    if (Math.abs(v) >= 1000)  return v.toFixed(0);
    if (Math.abs(v) >= 10)    return v.toFixed(1);
    return v.toFixed(3);
  }

  /** Draw the horizontal bar chart (Panel 3). */
  function drawBarChart(canvas, im, om) {
    if (!canvas) return;
    const W = canvas.clientWidth || canvas.width;
    canvas.width  = W;
    canvas.height = CANVAS_H;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, W, CANVAS_H);
    ctx.fillStyle = BG_COLOR;
    ctx.fillRect(0, 0, W, CANVAS_H);

    const metrics = [
      { label: 'Mean RMS',        inp: im.meanRms,              out: om.meanRms,              fmt: (v) => v.toFixed(4) },
      { label: 'Centroid (Hz)',   inp: im.meanSpectralCentroid,  out: om.meanSpectralCentroid,  fmt: (v) => v.toFixed(0) },
      { label: 'Rolloff (Hz)',    inp: im.meanSpectralRolloff,   out: om.meanSpectralRolloff,   fmt: (v) => v.toFixed(0) },
      { label: 'Flatness',        inp: im.meanSpectralFlatness,  out: om.meanSpectralFlatness,  fmt: (v) => v.toFixed(4) },
      { label: 'ZCR',             inp: im.meanZcr,               out: om.meanZcr,               fmt: (v) => v.toFixed(4) },
    ];

    const PAD_L = 90, PAD_R = 60, PAD_T = 10, PAD_B = 10;
    const plotW = W - PAD_L - PAD_R;
    const rowH  = (CANVAS_H - PAD_T - PAD_B) / metrics.length;
    const barH  = Math.min(10, rowH * 0.35);

    ctx.font = '10px Inter, sans-serif';

    metrics.forEach(({ label, inp, out, fmt }, idx) => {
      const maxVal = Math.max(Math.abs(inp), Math.abs(out), 1e-12);
      const y0 = PAD_T + idx * rowH + rowH / 2;

      // Metric name
      ctx.fillStyle = LABEL_COLOR;
      ctx.textAlign = 'right';
      ctx.fillText(label, PAD_L - 6, y0 + 3);

      // Input bar
      const inpW = (Math.abs(inp) / maxVal) * plotW;
      ctx.fillStyle = INPUT_COLOR;
      ctx.fillRect(PAD_L, y0 - barH - 1, inpW, barH);

      // Output bar
      const outW = (Math.abs(out) / maxVal) * plotW;
      ctx.fillStyle = OUTPUT_COLOR;
      ctx.fillRect(PAD_L, y0 + 1, outW, barH);

      // Value labels
      ctx.fillStyle = INPUT_COLOR;
      ctx.textAlign = 'left';
      ctx.fillText(fmt(inp), PAD_L + inpW + 3, y0 - 1);

      ctx.fillStyle = OUTPUT_COLOR;
      ctx.fillText(fmt(out), PAD_L + outW + 3, y0 + barH + 9);
    });

    // Legend
    ctx.font = '9px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = INPUT_COLOR;
    ctx.fillRect(PAD_L, CANVAS_H - PAD_B - 2, 10, 3);
    ctx.fillStyle = LABEL_COLOR;
    ctx.fillText('Input', PAD_L + 13, CANVAS_H - PAD_B + 2);
    ctx.fillStyle = OUTPUT_COLOR;
    ctx.fillRect(PAD_L + 55, CANVAS_H - PAD_B - 2, 10, 3);
    ctx.fillStyle = LABEL_COLOR;
    ctx.fillText('Output', PAD_L + 68, CANVAS_H - PAD_B + 2);
  }

  // -------------------------------------------------------------------------
  // Delta classification for Panel 4
  // -------------------------------------------------------------------------
  function deltaClass(metricKey, delta) {
    const POSITIVE = ['meanRms', 'meanSpectralFlatness', 'meanZcr'];
    const NEUTRAL   = ['meanSpectralCentroid', 'meanSpectralRolloff'];
    if (NEUTRAL.includes(metricKey)) return 'delta-neu';
    if (delta > 0 && POSITIVE.includes(metricKey)) return 'delta-pos';
    if (delta < 0 && POSITIVE.includes(metricKey)) return 'delta-neg';
    return 'delta-neu';
  }

  function fmtVal(key, v) {
    if (key === 'meanSpectralCentroid' || key === 'meanSpectralRolloff') return v.toFixed(0);
    return v.toFixed(4);
  }

  const DELTA_METRICS = [
    { key: 'meanRms',              label: 'Mean RMS' },
    { key: 'meanSpectralCentroid', label: 'Spectral Centroid (Hz)' },
    { key: 'meanSpectralRolloff',  label: 'Spectral Rolloff (Hz)' },
    { key: 'meanSpectralFlatness', label: 'Spectral Flatness' },
    { key: 'meanZcr',              label: 'Zero Crossing Rate' },
  ];

  // -------------------------------------------------------------------------
  // Reactive drawing: redraw whenever metrics or canvas refs change
  // -------------------------------------------------------------------------
  $effect(() => {
    if (!inputMetrics || !outputMetrics) return;
    const im = inputMetrics;
    const om = outputMetrics;

    drawLineChart(rmsCanvas,      im, om, (m) => m.rms,              'Amplitude');
    drawLineChart(centroidCanvas, im, om, (m) => m.spectralCentroid, 'Hz');
    drawBarChart(barCanvas, im, om);
  });

  // -------------------------------------------------------------------------
  // Resize listener
  // -------------------------------------------------------------------------
  $effect(() => {
    function onResize() {
      if (!inputMetrics || !outputMetrics) return;
      const im = inputMetrics;
      const om = outputMetrics;
      drawLineChart(rmsCanvas,      im, om, (m) => m.rms,              'Amplitude');
      drawLineChart(centroidCanvas, im, om, (m) => m.spectralCentroid, 'Hz');
      drawBarChart(barCanvas, im, om);
    }
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  });
</script>

<div class="analysis-root">
  <h2 class="section-title">Audio Analysis</h2>

  {#if loading}
    <p class="status-msg">Analysing audio…</p>
  {:else if error}
    <p class="status-msg error">{error}</p>
  {:else if inputMetrics && outputMetrics}
    <div class="analysis-grid">

      <!-- Panel 1: RMS over time -->
      <div class="analysis-panel">
        <h3>RMS Energy over Time</h3>
        <canvas bind:this={rmsCanvas} style="width:100%;height:{CANVAS_H}px;display:block;"></canvas>
      </div>

      <!-- Panel 2: Spectral Centroid over time -->
      <div class="analysis-panel">
        <h3>Spectral Centroid over Time</h3>
        <canvas bind:this={centroidCanvas} style="width:100%;height:{CANVAS_H}px;display:block;"></canvas>
      </div>

      <!-- Panel 3: Bar chart comparison -->
      <div class="analysis-panel">
        <h3>Scalar Metric Comparison</h3>
        <canvas bind:this={barCanvas} style="width:100%;height:{CANVAS_H}px;display:block;"></canvas>
        <div class="bar-legend">
          <span class="swatch input-swatch"></span><span>Input</span>
          <span class="swatch output-swatch"></span><span>Output</span>
        </div>
      </div>

      <!-- Panel 4: Delta summary table -->
      <div class="analysis-panel">
        <h3>Summary Delta (Output − Input)</h3>
        <table class="delta-table">
          <thead>
            <tr>
              <th>Metric</th>
              <th>Input</th>
              <th>Output</th>
              <th>Delta</th>
            </tr>
          </thead>
          <tbody>
            {#each DELTA_METRICS as { key, label }}
              {@const inp   = inputMetrics[key]}
              {@const out   = outputMetrics[key]}
              {@const delta = out - inp}
              {@const cls   = deltaClass(key, delta)}
              <tr>
                <td>{label}</td>
                <td>{fmtVal(key, inp)}</td>
                <td>{fmtVal(key, out)}</td>
                <td class={cls}>{delta >= 0 ? '+' : ''}{fmtVal(key, delta)}</td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

    </div>
  {/if}
</div>

<style>
  .analysis-root {
    margin-top: 2rem;
  }

  .section-title {
    font-size: 1rem;
    font-weight: 600;
    color: #a6adc8;
    margin: 0 0 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }

  .status-msg {
    color: #a6adc8;
    font-size: 0.85rem;
  }
  .status-msg.error {
    color: #f38ba8;
  }

  .analysis-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  @media (max-width: 600px) {
    .analysis-grid { grid-template-columns: 1fr; }
  }

  .analysis-panel {
    background: #1e1e2e;
    border-radius: 10px;
    padding: 1rem;
  }

  .analysis-panel h3 {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a6adc8;
    margin: 0 0 0.75rem;
  }

  .bar-legend {
    display: flex;
    gap: 0.75rem;
    align-items: center;
    margin-top: 0.5rem;
    font-size: 0.78rem;
    color: #a6adc8;
  }

  .swatch {
    display: inline-block;
    width: 12px;
    height: 3px;
    border-radius: 2px;
  }
  .input-swatch  { background: #89b4fa; }
  .output-swatch { background: #a6e3a1; }

  .delta-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }

  .delta-table th {
    text-align: left;
    color: #6c7086;
    font-weight: 500;
    padding: 0.3rem 0.5rem;
    border-bottom: 1px solid #313244;
  }

  .delta-table td {
    padding: 0.35rem 0.5rem;
    color: #cdd6f4;
    border-bottom: 1px solid #1e1e2e;
  }

  :global(.delta-pos) { color: #a6e3a1; }
  :global(.delta-neg) { color: #f38ba8; }
  :global(.delta-neu) { color: #cdd6f4; }
</style>
