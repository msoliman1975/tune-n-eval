// Typed client for the Studio API. Shapes follow contracts/*.schema.json.

export type Preset = { model: string; load_in_4bit: boolean; seq: number; chat_template: string; measured_vram_gb: number; tokens_per_s: number };
export type DatasetRef = { id: string; version: string; sha256: string; counts: Record<string, number>; created_at: string; path: string };
export type JobEvent = { job_id: string; type: string; ts: string; step?: number; total?: number; metrics?: Record<string, number>; message?: string };
export type ModelRef = { id: string; kind: string; base: string; uri?: string; ollama_tag?: string; lineage?: Record<string, string> };
export type RunSummary = {
  id: string; status: string; created_at: string | null; message: string | null; step: number; total: number | null;
  result: Record<string, number> | null; model_ref: ModelRef | null;
};
export type Estimate = { preset: string; vram_gb: number; vram_total_gb: number; fits: boolean; train_rows: number; est_minutes: number };
export type Score = { model: string; suite: string; metric: string; slice: { split: string; lang: string }; value: number; ci95: [number, number]; n: number };
export type Comparison = { a: string; b: string; suite: string; slice: { split: string; lang: string }; delta: number; ci95: [number, number]; verdict: string };
export type EvalReport = {
  id: string; status: string; models: ModelRef[]; suites: string[]; dataset: string; scores: Score[]; comparisons: Comparison[];
  judge: { model?: string; rubric?: string }; created_at: string; log_tail?: string[];
};
export type EvalSummary = { id: string; status: string; dataset: string | null; models: string[] | null; created_at: string | null };
export type Verdict = Record<"correctness" | "completeness" | "actionability" | "safety" | "language", number> & { rationale: string };
export type Example = { item: string; delta: number; base: { answer: string; verdict: Verdict }; tuned: { answer: string; verdict: Verdict } };
export type Source = { doc_id: string; title: string; lang: string; license: string; egypt_specific: boolean; download: string | null; parse: string | null; chars: number | null };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`/api${path}`, { headers: { "content-type": "application/json" }, ...init });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    throw new Error(body.detail ?? `${r.status} ${r.statusText}`);
  }
  return r.json();
}

export const api = {
  health: () => req<Record<string, boolean>>("/health"),
  presets: () => req<Record<string, Preset>>("/train/presets"),
  datasets: () => req<DatasetRef[]>("/data/datasets"),
  sources: () => req<Source[]>("/data/sources"),
  pipeline: () => req<Record<string, number>>("/data/pipeline"),
  runs: () => req<RunSummary[]>("/train/runs"),
  run: (id: string) => req<RunSummary & { events: JobEvent[] }>(`/train/runs/${id}`),
  estimate: (body: object) => req<Estimate>("/train/runs/estimate", { method: "POST", body: JSON.stringify(body) }),
  launch: (body: object) => req<{ id: string }>("/train/runs", { method: "POST", body: JSON.stringify(body) }),
  cancel: (id: string) => req(`/train/runs/${id}/cancel`, { method: "POST" }),
  evals: () => req<EvalSummary[]>("/eval/evaluations"),
  evalReport: (id: string) => req<EvalReport>(`/eval/evaluations/${id}`),
  examples: (id: string) => req<Example[]>(`/eval/evaluations/${id}/examples`),
  evaluate: (body: object) => req<{ id: string }>("/eval/evaluations", { method: "POST", body: JSON.stringify(body) }),
};
