// Typed client for the Studio API. Shapes follow contracts/*.schema.json and the service APIs.

export type Preset = { model: string; load_in_4bit: boolean; seq: number; chat_template: string; measured_vram_gb: number | null; tokens_per_s: number | null };
export type DatasetRef = {
  id: string; version: string; sha256: string; task: string; counts: Record<string, number>; created_at: string; path: string;
  splits: Record<string, string>; split_by: string; attachments?: string | null; stats?: { avg_chars: Record<string, number> };
  validation?: { checked: number; invalid: number; reasons: Record<string, number> }; source?: Record<string, unknown>;
};
export type JobEvent = { job_id: string; type: string; ts: string; step?: number; total?: number; metrics?: Record<string, number>; message?: string };
export type ModelRef = { id: string; kind: string; base: string; uri?: string; ollama_tag?: string; lineage?: Record<string, string> };
export type RunSummary = {
  id: string; status: string; created_at: string | null; message: string | null; step: number; total: number | null;
  result: Record<string, number> | null; model_ref: ModelRef | null; export: { status: string; ollama_tag?: string } | null; logs: string[];
};
export type Estimate = { preset: string; vram_gb: number | null; vram_total_gb: number; fits: boolean | null; train_rows: number; total_train_rows: number; avg_tokens: number; est_minutes: number | null };
export type Source = { doc_id: string; title: string; lang: string; license: string; egypt_specific: boolean; download: string | null; parse: string | null; chars: number | null };

// Tasks and drafts (Data service)
export type Task = {
  ref: string; id: string; version: string; name: string; description: string; system: string; user_template: string;
  inputs: string[]; target: string; context: Record<string, { type: string }>; attachments: { type: string; layout: string; key: string } | null;
  metrics: { available: string[]; default: string[] }; variables: string[];
};
export type SplitSpec = {
  mode: "column" | "random" | "single"; column?: string | null; values?: Record<string, string>; single?: string;
  validation_frac?: number; test_frac?: number; validation_n?: number; group_by?: string | null; seed?: number;
};
export type Mapping = { task: string; fields: Record<string, string>; id_column?: string | null; meta_columns: string[]; split: SplitSpec };
export type AttachmentStatus = { type?: string; key?: string; files: number; needed?: number; missing?: string[]; n_missing?: number; note?: string } | null;
export type Draft = {
  id: string; name: string; filename: string; n_rows: number; columns: string[]; sample: Record<string, unknown>[]; status: string;
  task?: string; mapping: Mapping | null; counts: Record<string, number>; dropped_unlabelled?: number; edits?: number;
  attachments: AttachmentStatus; frozen_as?: string; error?: string | null; progress?: { checked: number; total: number } | null;
  validation?: { checked: number; invalid: number; reasons: Record<string, number>; examples: { id: string; reason: string }[] };
  created_at: string; updated_at?: string;
};
export type Item = Record<string, unknown> & { id: string; split: string; edited?: boolean; meta?: unknown };
export type ColumnProfile = {
  name: string; type: string; role: string; nulls: number; distinct: number;
  chars?: { min: number; p50: number; p90: number; max: number; mean: number }; numbers?: { min: number; p50: number; max: number; mean: number };
  top?: [string, number][];
};
export type DatasetProfile = {
  columns: ColumnProfile[]; fields: { inputs: string[]; target: string; context: string[] }; attachment_key: string | null;
  tokens_hist: { categories: string[]; series: Record<string, number[]> } | null; tokens: Record<string, { p50: number; p90: number; max: number }>;
  coverage: { key: string; per_split: Record<string, { keys: number; top: [string, number][] }>; per_key: Record<string, Record<string, number>>; shared: Record<string, number> } | null;
  files: { name: string; bytes: number }[]; attachments: { path: string; files: number; bytes: number } | null; task: Task;
};
export type DatasetItemDetail = { row: Item; messages: { role: string; content: string }[] | null; applies: boolean; error: string | null; est_prompt_tokens: number };
export type DbInfo = {
  key: string; file: string; bytes: number; items: Record<string, number>;
  tables: { name: string; rows: number | null; columns: { name: string; type: string; pk: boolean; notnull: boolean }[]; foreign_keys: { column: string; ref_table: string; ref_column: string }[] }[];
};
export type TableRows = { table: string; total: number; columns: string[]; rows: (string | number | null)[][] };
export type PreviewItem = { id: string; split: string; messages?: { role: string; content: string }[]; target?: string; chars?: number; est_tokens?: number; error?: string };

// Registry
/** What deleting a fine-tune removes (DELETE /train/runs/{id}?dry_run=true). */
export type DeletionPlan = {
  run_id: string; name: string | null; status: string; folder: string; folder_bytes: number; registry_entry: boolean;
  evaluations: number; ollama_tags: string[]; ollama_kept: string[]; answer_cache_bytes: number; mlflow_run_id: string | null;
};

export type PromptTier = "small" | "medium" | "large";
export type ModelEntry = {
  id: string; name: string; kind: "hf_base" | "hf_adapter" | "ollama" | "api"; provider: string; hf_id?: string | null;
  adapter_uri?: string | null; ollama_tag?: string | null; api_model?: string | null; chat_template?: string | null;
  train_preset?: string | null; parent?: string | null; lineage: Record<string, string>; paid: boolean; options: Record<string, unknown>; prompt_tier?: PromptTier | null;
  tags: string[]; notes: string; evaluations: { eval_id: string; dataset: string; task?: string; split?: string; n?: number; metrics: Record<string, number>; created_at?: string }[];
  created_at: string; updated_at: string; servable: boolean; trainable: boolean;
};

// Eval
export type MetricInfo = { id: string; name: string; approach: string; kind: "rate" | "score" | "scale" | "seconds" | "category" | "value"; description: string; higher_is_better: boolean; judge: boolean; default?: boolean };
export type Slice = { key: string; value: string };
export type Score = { model: string; metric: string; slice: Slice; value: number; ci95: [number, number]; n: number; p50?: number; p90?: number };
export type Comparison = { a: string; b: string; metric: string; slice: Slice; delta: number; ci95: [number, number]; verdict: string; n: number };
export type EvalReport = {
  id: string; status: string; version: 2; task: string; dataset: string; dataset_path: string; splits: string[]; n_items: number; limit: number; seed: number;
  models: ModelEntry[]; metrics: string[]; metric_info: Record<string, MetricInfo>; scores: Score[]; comparisons: Comparison[];
  breakdowns: Record<string, Record<string, Record<string, number>>>;
  usage: Record<string, { input_tokens: number; output_tokens: number; errors: number; usd: number | null }>;
  judge: { model?: string; rubric?: string }; created_at: string;
};
export type EvalProgress = {
  id: string; status: string; log_tail: string[]; spec?: { models: ModelEntry[]; dataset: string; metrics: string[] };
  progress?: { stages: { stage: string; model?: string; message?: string; done?: number; total?: number }[]; log: string[] };
  estimate?: { n_items: number };
};
export type EvalSummary = {
  id: string; status: string; kind: "generic" | "mango"; dataset?: string | null; task?: string; models?: string[] | null;
  n_items?: number; primary_metric?: string; headline?: Record<string, number>; created_at: string | null;
};
export type ExecPreview = { columns: string[]; rows: unknown[][]; n_rows: number; truncated: boolean; error: string | null };
export type ItemResult = {
  model: string; output: string; error: string | null; latency_s: number; input_tokens: number | null; output_tokens: number | null;
  scores: Record<string, number | string>; details: { sql?: string; result?: ExecPreview; sql_failure_judge?: { category?: string; rationale?: string; error?: string }; llm_judge?: { score?: number; rationale?: string } };
};
export type EvalItem = {
  id: string; split: string; slices: Record<string, string>; inputs: Record<string, string>; reference: string;
  gold_result?: ExecPreview | null; results: Record<string, ItemResult>; messages?: { role: string; content: string }[];
  prompt_source?: "stored" | "re-rendered";
  as_sent?: Record<string, { format: "text" | "json" | "error"; label: string; text?: string; body?: unknown }>;
};
export type EvalEstimate = { n_items: number; task: string; models: Record<string, { kind: string; paid: boolean; est_usd?: number | null; est_input_tokens?: number; est_output_tokens?: number }> };
export type GenerateResult = {
  task: string; messages: { role: string; content: string }[];
  answers: { model: string; name: string; output?: string; sql?: string; result?: ExecPreview; latency_s?: number; input_tokens?: number; output_tokens?: number; error?: string | null }[];
};

// Mango evaluations (evaluate.py)
export type LegacyScore = { model: string; suite: string; metric: string; slice: { split: string; lang: string }; value: number; ci95: [number, number]; n: number };
export type LegacyComparison = { a: string; b: string; suite: string; slice: { split: string; lang: string }; delta: number; ci95: [number, number]; verdict: string };
export type LegacyReport = {
  id: string; status: string; models: ModelRef[]; suites: string[]; dataset: string; scores: LegacyScore[]; comparisons: LegacyComparison[];
  judge: { model?: string; rubric?: string }; created_at: string;
};
export type Verdict = Record<"correctness" | "completeness" | "actionability" | "safety" | "language", number> & { rationale: string };
export type Example = { item: string; delta: number; base: { answer: string; verdict: Verdict }; tuned: { answer: string; verdict: Verdict } };

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const isForm = init?.body instanceof FormData;
  const r = await fetch(`/api${path}`, { ...init, headers: isForm ? undefined : { "content-type": "application/json" } });
  if (!r.ok) {
    const body = await r.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d: { msg: string }) => d.msg).join("; ") : body.detail;
    throw new Error(detail ?? `${r.status} ${r.statusText}`);
  }
  return r.status === 204 ? (undefined as T) : r.json();
}
const post = <T,>(path: string, body?: unknown) => req<T>(path, { method: "POST", body: body instanceof FormData ? body : JSON.stringify(body ?? {}) });
const qs = (p: Record<string, string | number | undefined | null>) =>
  new URLSearchParams(Object.entries(p).filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => [k, String(v)])).toString();

export const api = {
  health: () => req<Record<string, boolean>>("/health"),
  // data
  tasks: () => req<Task[]>("/data/tasks"),
  datasets: () => req<DatasetRef[]>("/data/datasets"),
  dataset: (id: string, v: string) => req<DatasetRef>(`/data/datasets/${id}/${v}`),
  datasetProfile: (id: string, v: string) => req<DatasetProfile>(`/data/datasets/${id}/${v}/profile`),
  datasetItem: (id: string, v: string, split: string, item: string) =>
    req<DatasetItemDetail>(`/data/datasets/${id}/${v}/items/${encodeURIComponent(item)}?${qs({ split })}`),
  database: (id: string, v: string, key: string) => req<DbInfo>(`/data/datasets/${id}/${v}/attachments/${encodeURIComponent(key)}`),
  tableRows: (id: string, v: string, key: string, table: string, p: { offset?: number; limit?: number }) =>
    req<TableRows>(`/data/datasets/${id}/${v}/attachments/${encodeURIComponent(key)}/tables/${encodeURIComponent(table)}?${qs(p)}`),
  downloadUrl: (id: string, v: string, split: string, fmt: "jsonl" | "csv" | "parquet") => `/api/data/datasets/${id}/${v}/download?${qs({ split, fmt })}`,
  datasetItems: (id: string, v: string, p: { split: string; q?: string; offset?: number; limit?: number; field?: string; value?: string }) =>
    req<{ total: number; items: Item[] }>(`/data/datasets/${id}/${v}/items?${qs(p)}`),
  attachments: (id: string, v: string) => req<{ type: string | null; key?: string; keys: string[] }>(`/data/datasets/${id}/${v}/attachments`),
  context: (id: string, v: string, key: string) => req<Record<string, string>>(`/data/datasets/${id}/${v}/context?${qs({ key })}`),
  drafts: () => req<Draft[]>("/data/drafts"),
  draft: (id: string) => req<Draft>(`/data/drafts/${id}`),
  upload: (file: File, name: string) => { const f = new FormData(); f.append("file", file); if (name) f.append("name", name); return post<Draft>("/data/drafts", f); },
  draftFromDataset: (dataset: string) => post<Draft>("/data/drafts/from-dataset", { dataset }),
  deleteDraft: (id: string) => req(`/data/drafts/${id}`, { method: "DELETE" }),
  columnValues: (id: string, col: string) => req<{ distinct: number; values: Record<string, number> }>(`/data/drafts/${id}/columns/${encodeURIComponent(col)}`),
  setMapping: (id: string, m: Mapping) => req<Draft>(`/data/drafts/${id}/mapping`, { method: "PUT", body: JSON.stringify(m) }),
  attachZip: (id: string, file: File) => { const f = new FormData(); f.append("file", file); return post<Draft>(`/data/drafts/${id}/attachments`, f); },
  attachPath: (id: string, path: string) => post<Draft>(`/data/drafts/${id}/attachments/path`, { path }),
  draftItems: (id: string, p: { split?: string; q?: string; offset?: number; limit?: number }) =>
    req<{ total: number; items: Item[] }>(`/data/drafts/${id}/items?${qs(p)}`),
  editItem: (id: string, item: string, fields: Record<string, unknown>) =>
    req<Item>(`/data/drafts/${id}/items/${encodeURIComponent(item)}`, { method: "PATCH", body: JSON.stringify(fields) }),
  deleteItem: (id: string, item: string) => req(`/data/drafts/${id}/items/${encodeURIComponent(item)}`, { method: "DELETE" }),
  preview: (id: string, n = 3) => req<PreviewItem[]>(`/data/drafts/${id}/preview?n=${n}`),
  freeze: (id: string, body: { id: string; version: string; drop_invalid: boolean; check_gold: boolean }) => post(`/data/drafts/${id}/freeze`, body),
  sources: () => req<Source[]>("/data/sources"),
  pipeline: () => req<Record<string, number>>("/data/pipeline"),
  // registry
  models: () => req<ModelEntry[]>("/registry/models"),
  model: (id: string) => req<ModelEntry>(`/registry/models/${id}`),
  registerModel: (m: Partial<ModelEntry>) => post<ModelEntry>("/registry/models", m),
  updateModel: (id: string, fields: Partial<ModelEntry>) => req<ModelEntry>(`/registry/models/${id}`, { method: "PATCH", body: JSON.stringify(fields) }),
  deleteModel: (id: string) => req(`/registry/models/${id}`, { method: "DELETE" }),
  // train
  presets: () => req<Record<string, Preset>>("/train/presets"),
  runs: () => req<RunSummary[]>("/train/runs"),
  run: (id: string) => req<RunSummary & { events: JobEvent[] }>(`/train/runs/${id}`),
  estimate: (body: object) => post<Estimate>("/train/runs/estimate", body),
  launch: (body: object) => post<{ id: string }>("/train/runs", body),
  cancel: (id: string) => post(`/train/runs/${id}/cancel`),
  deletionPlan: (id: string) => req<DeletionPlan>(`/train/runs/${id}?dry_run=true`, { method: "DELETE" }),
  deleteRun: (id: string) => req<DeletionPlan & { deleted: boolean; warnings: string[] }>(`/train/runs/${id}`, { method: "DELETE" }),
  exportRun: (id: string) => post(`/train/runs/${id}/export`),
  // eval
  metrics: (task: string) => req<{ approaches: Record<string, string>; metrics: MetricInfo[] }>(`/eval/metrics?${qs({ task })}`),
  evals: () => req<EvalSummary[]>("/eval/evaluations"),
  evalReport: (id: string) => req<EvalReport | EvalProgress | LegacyReport>(`/eval/evaluations/${id}`),
  evalItems: (id: string, p: { model?: string; metric?: string; outcome?: string; slice?: string; q?: string; offset?: number; limit?: number }) =>
    req<{ total: number; metric: string; items: EvalItem[] }>(`/eval/evaluations/${id}/items?${qs(p)}`),
  evalItem: (id: string, item: string) => req<EvalItem>(`/eval/evaluations/${id}/items/${encodeURIComponent(item)}`),
  evalEstimate: (body: object) => post<EvalEstimate>("/eval/evaluations/estimate", body),
  evaluate: (body: object) => post<{ id: string }>("/eval/evaluations", body),
  cancelEval: (id: string) => post(`/eval/evaluations/${id}/cancel`),
  examples: (id: string) => req<Example[]>(`/eval/evaluations/${id}/examples`),
  generate: (body: object) => post<GenerateResult>("/eval/generate", body),
};

/** 0-1 rates and scores as %, 1-5 scales to 2 decimals, seconds with an s. */
export function fmtMetric(kind: string | undefined, v: number | undefined | null) {
  if (v === undefined || v === null || Number.isNaN(v)) return "—";
  if (kind === "rate" || kind === "score") return `${(v * 100).toFixed(1)}%`;
  if (kind === "seconds") return `${v.toFixed(2)} s`;
  return v.toFixed(2);
}
