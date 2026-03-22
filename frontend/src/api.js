// src/api.js
const BASE = "http://localhost:8000";

export async function transformLatent(file) {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/transform`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json(); // { uid, synth_url, euler_url, heun_url, sample_rate }
}

export function audioUrl(path) {
  return `${BASE}${path}`;
}