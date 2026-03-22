<!-- src/lib/components/audiocard.svelte -->
<script>
  let { label, src, filename } = $props();
</script>

<div class="card">
  <p class="label">{label}</p>
  <audio controls {src}></audio>
  <a class="dl" href={src} download={filename}>↓ Download</a>
</div>

<style>
  .card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .label { color: #cdd6f4; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; margin: 0; }
  audio { width: 100%; accent-color: #cba6f7; }
  .dl { color: #89b4fa; font-size: 0.85rem; text-decoration: none; }
  .dl:hover { text-decoration: underline; }
</style>