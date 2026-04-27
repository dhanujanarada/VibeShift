// src/api.js
const BASE = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function transformLatent(file, timeoutMs = 180_000) {
  const form = new FormData();
  form.append("file", file);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${BASE}/transform`, {
      method: "POST",
      body: form,
      signal: controller.signal,
    });
    if (!res.ok) throw new Error(await res.text());
    return res.json(); // { uid, synth_url, euler_url, heun_url, sample_rate }
  } catch (e) {
    if (e.name === "AbortError") throw new Error("Request timed out — the server may be waking up, please try again.");
    throw e;
  } finally {
    clearTimeout(timer);
  }
}

export function audioUrl(path) {
  return `${BASE}${path}`;
}