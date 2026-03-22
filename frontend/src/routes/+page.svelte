<!-- src/routes/+page.svelte -->
<script>
  import AudioCard from "$lib/components/audiocard.svelte";
  import MelSpectrogram from "$lib/components/MelSpectrogram.svelte";
  import AudioAnalysis from "$lib/components/AudioAnalysis.svelte";
  import { transformLatent, audioUrl } from "../api.js";

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

    <AudioAnalysis
      inputSrc={audioUrl(result.input_url)}
      outputSrc={audioUrl(result.output_url)}
    />
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
</style>