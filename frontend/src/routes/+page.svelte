<!-- src/routes/+page.svelte -->
<script>
  import AudioCard from "$lib/components/audiocard.svelte";
  import MelSpectrogram from "$lib/components/MelSpectrogram.svelte";
  import { transformLatent, audioUrl } from "../api.js";

  import radarImg from "$lib/assets/radar_similarity.png";
  import barImg from "$lib/assets/bar_distance_metrics.png";
  import mfccImg from "$lib/assets/mfcc_per_band_mae.png";
  import pcaImg from "$lib/assets/pca_scatter.png";
  import fadData from "$lib/assets/fad_scores.json";
  import summaryRaw from "$lib/assets/summary_metrics.csv?raw";

  const evalPlots = [
    { src: radarImg, caption: "Similarity Metrics" },
    { src: barImg, caption: "Distance Metrics" },
    { src: mfccImg, caption: "MFCC Per-Band MAE" },
    { src: pcaImg, caption: "MFCC Space — PCA" },
  ];

  // Parse summary CSV
  function parseCSV(raw) {
    const lines = raw.trim().split(/\r?\n/);
    const headers = lines[0].split(",");
    const rows = lines.slice(1).map(l => l.split(","));
    return { headers, rows };
  }
  const { headers: summaryHeaders, rows: summaryRows } = parseCSV(summaryRaw);

  // FAD scores as sorted entries
  const fadEntries = Object.entries(fadData).sort(([a], [b]) => a.localeCompare(b));

  let file = $state(null);
  let status = $state("idle"); // idle | loading | done | error
  let result = $state(null);
  let errorMsg = $state("");

  function onFileChange(e) {
    file = e.target.files[0] ?? null;
    status = "idle";
    result = null;
  }

  async function onGenerate() {
    if (!file) return;
    status = "loading";
    errorMsg = "";
    try {
      result = await transformLatent(file);
      status = "done";
    } catch (e) {
      errorMsg = e.message;
      status = "error";
    }
  }
</script>

<main>
  <header>
    <h1>VibeShift</h1>
    <p class="sub">Audio → MIDI → Synth → Genre Transfer</p>
  </header>

  <section class="upload-panel">
    <label class="file-drop">
      <input type="file" accept=".mp3,.wav,.flac,.ogg,.m4a" onchange={onFileChange} />
      {#if file}
        <span class="filename">{file.name}</span>
      {:else}
        <span class="hint">Drop an audio file or click to browse</span>
      {/if}
    </label>

    <button
      class="generate-btn"
      disabled={!file || status === "loading"}
      onclick={onGenerate}
    >
      {status === "loading" ? "Generating…" : "Generate"}
    </button>
  </section>

  {#if status === "error"}
    <p class="error">{errorMsg}</p>
  {/if}

  {#if status === "done" && result}
    <section class="results">
      <AudioCard
        label="Input"
        src={audioUrl(result.input_url)}
        filename="input.wav"
      />
      <AudioCard
        label="MIDI Synth"
        src={audioUrl(result.synth_url)}
        filename="synth.wav"
      />
      <AudioCard
        label="Output"
        src={audioUrl(result.output_url)}
        filename="output.wav"
      />
    </section>

    <section class="mels">
      <h2>Mel Spectrograms</h2>
      <div class="mel-grid">
        <MelSpectrogram src={audioUrl(result.input_url)}  label="Input" />
        <MelSpectrogram src={audioUrl(result.synth_url)}  label="MIDI Synth" />
        <MelSpectrogram src={audioUrl(result.output_url)} label="Output" />
      </div>
    </section>
  {/if}

  {#if evalPlots.length}
    <section class="eval">
      <h2 class="eval-heading">Evaluation Results</h2>

      <div class="eval-col">
        {#each evalPlots as plot}
          <figure class="eval-card">
            <img src={plot.src} alt={plot.caption} />
            <figcaption>{plot.caption}</figcaption>
          </figure>
        {/each}
      </div>

      {#if fadEntries.length}
        <h3 class="eval-sub-heading">FAD Scores</h3>
        <ul class="fad-list">
          {#each fadEntries as [pair, score]}
            <li class="fad-item">
              <span class="fad-pair">{pair.replace(/_/g, " ")}</span>
              <span class="fad-score">{score !== null ? score.toFixed(4) : "—"}</span>
            </li>
          {/each}
        </ul>
      {/if}

      {#if summaryRows.length}
        <h3 class="eval-sub-heading">Summary Metrics</h3>
        <div class="table-wrap">
          <table class="summary-table">
            <thead>
              <tr>
                {#each summaryHeaders as h}
                  <th>{h}</th>
                {/each}
              </tr>
            </thead>
            <tbody>
              {#each summaryRows as row}
                <tr>
                  {#each row as cell, i}
                    <td class={i === 0 ? "pair-cell" : ""}>{isNaN(Number(cell)) || cell.trim() === "" ? cell : Number(cell).toFixed(4)}</td>
                  {/each}
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}
    </section>
  {/if}
</main>

<style>
  :global(body) { background: #11111b; color: #cdd6f4; font-family: "Inter", sans-serif; margin: 0; }
  main { max-width: 720px; margin: 0 auto; padding: 3rem 1.5rem; }
  header { margin-bottom: 2.5rem; }
  h1 { font-size: 2rem; font-weight: 700; margin: 0; color: #cba6f7; }
  .sub { color: #6c7086; margin: 0.3rem 0 0; }

  .upload-panel { display: flex; flex-direction: column; gap: 1rem; }
  .file-drop {
    display: flex; align-items: center; justify-content: center;
    border: 2px dashed #45475a; border-radius: 12px;
    padding: 2.5rem; cursor: pointer; transition: border-color 0.2s;
  }
  .file-drop:hover { border-color: #cba6f7; }
  .file-drop input { display: none; }
  .hint { color: #585b70; }
  .filename { color: #a6e3a1; font-weight: 500; }

  .generate-btn {
    background: #cba6f7; color: #11111b; border: none;
    border-radius: 8px; padding: 0.75rem 2rem;
    font-size: 1rem; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s;
  }
  .generate-btn:disabled { opacity: 0.4; cursor: not-allowed; }

  .error { color: #f38ba8; margin-top: 1rem; }

  .results { display: flex; flex-direction: column; gap: 1rem; margin-top: 2rem; }
  code { font-family: monospace; color: #89b4fa; }

  .mels h2 {
    font-size: 1rem;
    font-weight: 600;
    color: #a6adc8;
    margin: 2rem 0 1rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .mel-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
  }
  @media (max-width: 600px) {
    .mel-grid { grid-template-columns: 1fr; }
  }

  /* ── Evaluation Results ─────────────────────────────── */
  .eval { margin-top: 3rem; }
  .eval-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #cba6f7;
    margin: 0 0 1.5rem;
  }
  .eval-sub-heading {
    font-size: 0.85rem;
    font-weight: 600;
    color: #a6adc8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2rem 0 0.75rem;
  }
  .eval-col {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
  }
  .eval-card {
    margin: 0;
    border: 1px solid #45475a;
    border-radius: 12px;
    overflow: hidden;
    background: #181825;
  }
  .eval-card img {
    width: 100%;
    display: block;
  }
  .eval-card figcaption {
    padding: 0.5rem 0.75rem;
    color: #6c7086;
    font-size: 0.85rem;
    text-align: center;
  }

  /* FAD list */
  .fad-list {
    list-style: none;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  .fad-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1rem;
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 8px;
    font-size: 0.9rem;
  }
  .fad-pair { color: #cdd6f4; text-transform: capitalize; }
  .fad-score { color: #cba6f7; font-variant-numeric: tabular-nums; font-weight: 600; }

  /* Summary table */
  .table-wrap {
    overflow-x: auto;
    border: 1px solid #45475a;
    border-radius: 12px;
  }
  .summary-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.8rem;
    white-space: nowrap;
  }
  .summary-table thead { background: #1e1e2e; }
  .summary-table th {
    padding: 0.55rem 0.8rem;
    color: #a6adc8;
    font-weight: 600;
    text-align: left;
    border-bottom: 1px solid #45475a;
  }
  .summary-table td {
    padding: 0.5rem 0.8rem;
    color: #cdd6f4;
    border-bottom: 1px solid #313244;
  }
  .summary-table tbody tr:last-child td { border-bottom: none; }
  .summary-table tbody tr:hover td { background: #1e1e2e; }
  .pair-cell { color: #cba6f7; font-weight: 600; }
</style>