import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type ModelEntry } from "../api";
import { Badge, Button, Card, Code, ErrorNote, Field, inputCls, Tabs, when } from "../ui";

const KIND: Record<string, string> = { hf_base: "base (local GPU)", hf_adapter: "fine-tuned", ollama: "Ollama", api: "API" };

export function where(m: ModelEntry) {
  return m.kind === "api" ? `${m.provider} · ${m.api_model}` : m.kind === "ollama" ? m.ollama_tag
    : m.kind === "hf_adapter" ? `adapter on ${m.parent ?? m.hf_id}` : m.hf_id;
}

const SHOW: [string, string][] = [["all", "All"], ["hf_adapter", "Fine-tuned"], ["hf_base", "Base"], ["ollama", "Ollama"], ["api", "API"]];

export function Models() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const groups = ["hf_base", "hf_adapter", "ollama", "api"] as const;
  const [show, setShow] = useState("all");
  const [q, setQ] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const all = models.data ?? [];
  const allTags = [...new Set(all.flatMap((m) => m.tags))].sort();
  const text = (m: ModelEntry) => [m.name, m.id, m.notes, ...m.tags, m.lineage?.dataset, m.parent].join(" ").toLowerCase();
  const shown = groups.flatMap((g) => all.filter((m) => m.kind === g))
    .filter((m) => show === "all" || m.kind === show)
    .filter((m) => tags.every((t) => m.tags.includes(t)))
    .filter((m) => !q || text(m).includes(q.toLowerCase()));
  const toggle = (t: string) => setTags(tags.includes(t) ? tags.filter((x) => x !== t) : [...tags, t]);
  return (
    <div className="grid gap-6">
      <Card title="Model registry">
        <p className="mb-3 text-sm text-muted">Every model the platform can train or evaluate. Fine-tuned runs register themselves when training succeeds; every evaluation is recorded on the models it measured. Name and tag a model on its page.</p>
        <ErrorNote error={models.error} />
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <Tabs value={show} options={SHOW.map(([k, l]) => [k, `${l} ${k === "all" ? all.length : all.filter((m) => m.kind === k).length}`] as [string, string])} onChange={setShow} />
          <input className={`${inputCls} w-full sm:w-64`} placeholder="Search name, id, tag, dataset…" value={q} onChange={(e) => setQ(e.target.value)} />
        </div>
        {allTags.length > 0 && (
          <div className="mb-3 flex flex-wrap items-center gap-1.5">
            <span className="font-mono text-xs uppercase text-muted">Tags</span>
            {allTags.map((t) => <TagChip key={t} tag={t} active={tags.includes(t)} onClick={() => toggle(t)} />)}
            {tags.length > 0 && <button className="text-xs text-accent" onClick={() => setTags([])}>clear</button>}
          </div>
        )}
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-sm">
            <thead className="text-left font-mono text-xs uppercase text-muted">
              <tr><th className="py-2">Model</th><th>Kind</th><th>Trained on</th><th>Served from</th><th>Can</th><th>Latest evaluation</th></tr>
            </thead>
            <tbody>
              {shown.map((m) => {
                const last = m.evaluations.at(-1);
                return (
                  <tr key={m.id} className="border-t border-line align-top">
                    <td className="py-2 pr-3">
                      <a className="text-accent" href={`#/models/${m.id}`}>{m.name}</a>
                      <div className="font-mono text-xs text-muted">{m.id}</div>
                      {m.tags.length > 0 && <div className="mt-1 flex flex-wrap gap-1">{m.tags.map((t) => <TagChip key={t} tag={t} active={tags.includes(t)} onClick={() => toggle(t)} />)}</div>}
                    </td>
                    <td>{KIND[m.kind]}{m.paid && <> <Badge value="paid" /></>}</td>
                    <td className="text-xs">{m.kind === "hf_adapter" ? <>
                      <span className="font-mono">{String(m.lineage?.dataset ?? "?")}</span>
                      {m.lineage?.train_fraction ? <span className="text-muted"> · {Math.round(Number(m.lineage.train_fraction) * 100)}%</span> : null}
                      <div className="text-muted">from {m.parent ?? m.hf_id} · {when(m.created_at)}</div>
                    </> : <span className="text-muted">—</span>}</td>
                    <td className="max-w-64 break-all font-mono text-xs">{m.kind === "hf_adapter" ? "local GPU (adapter)" : where(m)}{m.kind !== "ollama" && m.ollama_tag ? <div className="text-muted">ollama {m.ollama_tag}</div> : null}</td>
                    <td className="text-xs">{[m.trainable && "fine-tune", m.servable && "playground", "evaluate"].filter(Boolean).join(" · ")}</td>
                    <td className="text-xs">{last ? <a className="text-accent" href={`#/evals/${last.eval_id}`}>{last.dataset}: {Object.entries(last.metrics).slice(0, 2).map(([k, v]) => `${k} ${k === "latency" ? v.toFixed(2) + "s" : (v * 100).toFixed(1) + "%"}`).join(", ")}</a> : <span className="text-muted">—</span>}</td>
                  </tr>
                );
              })}
              {models.data && shown.length === 0 && <tr><td colSpan={6} className="py-6 text-center text-muted">No model matches.</td></tr>}
            </tbody>
          </table>
        </div>
      </Card>
      <RegisterCard />
    </div>
  );
}

/** A tag; clickable ones filter the list. */
export function TagChip({ tag, active = false, onClick, onRemove }: { tag: string; active?: boolean; onClick?: () => void; onRemove?: () => void }) {
  const look = `inline-flex items-center gap-1 rounded-full border px-2 py-0.5 font-mono text-xs ${active ? "border-accent bg-accent-soft text-accent" : "border-line text-muted"}`;
  if (onRemove) return <span className={look}>{tag}<button type="button" aria-label={`remove tag ${tag}`} className="hover:text-warn" onClick={onRemove}>×</button></span>;
  return onClick ? <button type="button" className={`${look} hover:border-accent`} onClick={onClick}>{tag}</button> : <span className={look}>{tag}</span>;
}

/** Tags as chips; type and press Enter or comma to add one. */
export function TagEditor({ tags, onChange, suggestions = [] }: { tags: string[]; onChange: (t: string[]) => void; suggestions?: string[] }) {
  const [draft, setDraft] = useState("");
  const add = (raw: string) => {
    const t = raw.trim().toLowerCase();
    if (t && !tags.includes(t)) onChange([...tags, t]);
    setDraft("");
  };
  return (
    <div className={`${inputCls} flex flex-wrap items-center gap-1.5`}>
      {tags.map((t) => <TagChip key={t} tag={t} onRemove={() => onChange(tags.filter((x) => x !== t))} />)}
      <input className="min-w-24 flex-1 bg-transparent outline-none" value={draft} list="known-tags" placeholder={tags.length ? "" : "e.g. best, baseline, prompt-v2"}
        onChange={(e) => e.target.value.endsWith(",") ? add(e.target.value.slice(0, -1)) : setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") { e.preventDefault(); add(draft); }
          if (e.key === "Backspace" && !draft && tags.length) onChange(tags.slice(0, -1));
        }}
        onBlur={() => draft && add(draft)} />
      <datalist id="known-tags">{suggestions.filter((s) => !tags.includes(s)).map((s) => <option key={s} value={s} />)}</datalist>
    </div>
  );
}

const size = (b: number) => b >= 1024 ** 3 ? `${(b / 1024 ** 3).toFixed(2)} GB` : b >= 1024 ** 2 ? `${(b / 1024 ** 2).toFixed(1)} MB` : `${Math.max(1, Math.round(b / 1024))} KB`;

/** Deletes a fine-tune (its run) everywhere, after showing exactly what goes. */
export function DeleteFineTune({ runId, onDeleted }: { runId: string; onDeleted: () => void }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const plan = useQuery({ queryKey: ["deletion", runId], queryFn: () => api.deletionPlan(runId), enabled: open, staleTime: 0 });
  const del = useMutation({
    mutationFn: () => api.deleteRun(runId),
    onSuccess: (r) => {
      qc.invalidateQueries({ queryKey: ["models"] }); qc.invalidateQueries({ queryKey: ["runs"] });
      if (r.warnings.length) alert(`Deleted, with warnings:\n${r.warnings.join("\n")}`);
      onDeleted();
    },
  });
  const p = plan.data;
  return (
    <Card title="Delete">
      {!open ? (
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted">Removes this fine-tune from the platform and frees its disk space. Evaluation reports that measured it are kept.</p>
          <Button tone="danger" onClick={() => setOpen(true)}>Delete fine-tune…</Button>
        </div>
      ) : (
        <div className="grid gap-3 text-sm">
          <ErrorNote error={plan.error ?? del.error} />
          {!p ? (!plan.error && <p className="text-muted">Checking what it left behind…</p>) : <>
            <p>Deleting <b>{p.name ?? p.run_id}</b> removes, permanently:</p>
            <ul className="grid list-disc gap-1 pl-5">
              <li>the run folder <span className="font-mono text-xs break-all">{p.folder}</span> (adapter, checkpoints, logs{p.ollama_tags.length ? ", GGUF" : ""}): <b className="num">{size(p.folder_bytes)}</b></li>
              {p.ollama_tags.map((t) => <li key={t}>the Ollama model <span className="font-mono text-xs">{t}</span></li>)}
              {p.registry_entry && <li>its Model Registry entry (name, tags, notes)</li>}
              {p.answer_cache_bytes > 0 && <li>its cached evaluation answers ({size(p.answer_cache_bytes)})</li>}
              {p.mlflow_run_id && <li>its MLflow run (moved to MLflow's deleted runs)</li>}
            </ul>
            {p.ollama_kept.length > 0 && <p className="text-muted">Kept: Ollama model {p.ollama_kept.join(", ")}, which another registry entry uses.</p>}
            <p className="text-muted">{p.evaluations ? `Its ${p.evaluations} evaluation report${p.evaluations > 1 ? "s stay" : " stays"}, with answers and scores, under Evaluations.` : "It has no evaluations."}</p>
            <div className="flex flex-wrap gap-2">
              <Button tone="danger" disabled={del.isPending} onClick={() => del.mutate()}>{del.isPending ? "Deleting…" : `Delete permanently (${size(p.folder_bytes)})`}</Button>
              <Button tone="plain" onClick={() => setOpen(false)}>Keep it</Button>
            </div>
          </>}
        </div>
      )}
    </Card>
  );
}

function RegisterCard() {
  const qc = useQueryClient();
  const [kind, setKind] = useState<"ollama" | "api" | "hf_base">("ollama");
  const [f, setF] = useState({ id: "", name: "", ollama_tag: "", provider: "anthropic", api_model: "", hf_id: "", notes: "" });
  const set = (k: keyof typeof f) => (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => setF({ ...f, [k]: e.target.value });
  const go = useMutation({
    mutationFn: () => api.registerModel({
      id: f.id, name: f.name || f.id, kind, notes: f.notes,
      ...(kind === "ollama" ? { ollama_tag: f.ollama_tag } : kind === "api" ? { provider: f.provider, api_model: f.api_model, paid: true } : { hf_id: f.hf_id }),
    }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); setF({ ...f, id: "", name: "" }); },
  });
  return (
    <Card title="Register a model">
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Field label="Kind">
          <select className={inputCls} value={kind} onChange={(e) => setKind(e.target.value as typeof kind)}>
            <option value="ollama">Ollama model (local)</option><option value="api">Hosted API model (paid)</option><option value="hf_base">Hugging Face checkpoint (eval only)</option>
          </select>
        </Field>
        <Field label="Id" hint="lowercase, digits, . _ -"><input className={inputCls} value={f.id} onChange={set("id")} placeholder="llama-3.2-3b" /></Field>
        <Field label="Name"><input className={inputCls} value={f.name} onChange={set("name")} /></Field>
        {kind === "ollama" && <Field label="Ollama tag"><input className={inputCls} value={f.ollama_tag} onChange={set("ollama_tag")} placeholder="llama3.2:3b" /></Field>}
        {kind === "api" && <>
          <Field label="Provider"><select className={inputCls} value={f.provider} onChange={set("provider")}><option value="anthropic">Anthropic</option><option value="google">Google (Gemini)</option></select></Field>
          <Field label="Provider model id"><input className={inputCls} value={f.api_model} onChange={set("api_model")} placeholder="claude-haiku-5-5" /></Field>
        </>}
        {kind === "hf_base" && <Field label="Hugging Face id"><input className={inputCls} value={f.hf_id} onChange={set("hf_id")} placeholder="org/model" /></Field>}
        <Field label="Notes"><input className={inputCls} value={f.notes} onChange={set("notes")} /></Field>
      </div>
      <div className="mt-4"><Button disabled={!f.id || go.isPending} onClick={() => go.mutate()}>Register</Button></div>
      <ErrorNote error={go.error} />
    </Card>
  );
}

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return <><dt className="font-mono text-xs uppercase text-muted">{k}</dt><dd className="break-all font-mono text-xs">{children}</dd></>;
}

export function ModelDetail({ id }: { id: string }) {
  const qc = useQueryClient();
  const m = useQuery({ queryKey: ["model", id], queryFn: () => api.model(id) });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const [edit, setEdit] = useState<{ name?: string; tags?: string[]; notes?: string; options?: string }>({});
  const save = useMutation({
    mutationFn: () => api.updateModel(id, {
      ...(edit.name !== undefined ? { name: edit.name } : {}), ...(edit.tags !== undefined ? { tags: edit.tags } : {}),
      ...(edit.notes !== undefined ? { notes: edit.notes } : {}), ...(edit.options !== undefined ? { options: JSON.parse(edit.options || "{}") } : {}),
    }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["model", id] }); qc.invalidateQueries({ queryKey: ["models"] }); setEdit({}); },
  });
  const del = useMutation({ mutationFn: () => api.deleteModel(id), onSuccess: () => { qc.invalidateQueries({ queryKey: ["models"] }); location.hash = "#/models"; } });
  const e = m.data;
  if (!e) return <Card><ErrorNote error={m.error} /></Card>;
  const lineage = Object.entries(e.lineage ?? {});
  const changed = Object.keys(edit).length > 0;
  const fineTune = e.kind === "hf_adapter" && !!e.lineage?.run_id;
  const known = [...new Set((models.data ?? []).flatMap((x) => x.tags))].sort();
  return (
    <div className="grid gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <a href="#/models" className="text-sm text-muted">← Models</a>
          <h1 className="text-2xl font-bold">{e.name}</h1>
          <p className="break-all font-mono text-sm text-muted">{e.id} · {KIND[e.kind]} · {where(e)}</p>
          {e.tags.length > 0 && <div className="mt-2 flex flex-wrap gap-1">{e.tags.map((t) => <TagChip key={t} tag={t} />)}</div>}
        </div>
        <div className="flex gap-2">
          {e.trainable && <a className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-accent-soft" href={`#/training?model=${e.id}`}>Fine-tune</a>}
          <a className="rounded-md bg-accent px-3 py-1.5 text-sm text-bg" href={`#/evals/new?models=${[e.parent, e.id].filter(Boolean).join(",")}`}>Evaluate{e.parent ? " vs its base" : ""}</a>
        </div>
      </div>
      <div className="grid gap-6 md:grid-cols-2">
        <Card title="Details">
          <dl className="grid grid-cols-[8rem_1fr] gap-x-3 gap-y-1 text-sm">
            {([["kind", KIND[e.kind]], ["provider", e.provider], ["checkpoint", e.hf_id], ["adapter", e.adapter_uri], ["ollama tag", e.ollama_tag],
               ["api model", e.api_model], ["train preset", e.train_preset], ["paid", e.paid ? "yes" : "no"], ["registered", when(e.created_at)]] as [string, unknown][])
              .filter(([, v]) => v).map(([k, v]) => <Row key={k} k={k}>{String(v)}</Row>)}
            {e.parent && <Row k="parent"><a className="text-accent" href={`#/models/${e.parent}`}>{e.parent}</a></Row>}
          </dl>
          {lineage.length > 0 && (
            <>
              <h3 className="mt-4 mb-1 font-bold">Lineage</h3>
              <dl className="grid grid-cols-[8rem_1fr] gap-x-3 gap-y-1 text-sm">
                {lineage.map(([k, v]) => <Row key={k} k={k}>{k === "run_id" ? <a className="text-accent" href={`#/runs/${v}`}>{String(v)}</a> : String(v)}</Row>)}
              </dl>
            </>
          )}
        </Card>
        <Card title="Name, tags and notes">
          <div className="grid gap-3">
            <Field label="Name" hint="shown wherever models are picked or compared; unique">
              <input className={inputCls} value={edit.name ?? e.name} onChange={(ev) => setEdit({ ...edit, name: ev.target.value })} />
            </Field>
            <Field label="Tags" hint="Enter or comma adds one; filter by tags on the Models page">
              <TagEditor tags={edit.tags ?? e.tags} onChange={(tags) => setEdit({ ...edit, tags })} suggestions={known} />
            </Field>
            <Field label="Notes"><textarea rows={3} className={inputCls} value={edit.notes ?? e.notes} onChange={(ev) => setEdit({ ...edit, notes: ev.target.value })} /></Field>
            <Field label="Request options (JSON)" hint={e.kind === "api" ? 'e.g. {"effort": "low"} for Claude to cut thinking cost' : "provider-specific"}>
              <textarea rows={2} className={`${inputCls} font-mono`} value={edit.options ?? JSON.stringify(e.options)} onChange={(ev) => setEdit({ ...edit, options: ev.target.value })} /></Field>
          </div>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button disabled={!changed || save.isPending || edit.name?.trim() === ""} onClick={() => save.mutate()}>Save</Button>
            {changed && <Button tone="plain" onClick={() => setEdit({})}>Discard</Button>}
            {!fineTune && <Button tone="danger" onClick={() => confirm(`Remove ${e.id} from the registry? Nothing on disk or in Ollama is deleted.`) && del.mutate()}>Remove from registry</Button>}
          </div>
          <ErrorNote error={save.error ?? del.error} />
        </Card>
      </div>
      <Card title={`Evaluations (${e.evaluations.length})`}>
        {e.evaluations.length === 0 ? <p className="text-sm text-muted">Not evaluated yet.</p> : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-sm">
              <thead className="text-left font-mono text-xs uppercase text-muted"><tr><th className="py-2">Evaluation</th><th>Dataset</th><th>Items</th><th>Scores</th></tr></thead>
              <tbody>
                {[...e.evaluations].reverse().map((ev) => (
                  <tr key={ev.eval_id} className="border-t border-line">
                    <td className="py-2"><a className="text-accent" href={`#/evals/${ev.eval_id}`}>{ev.eval_id}</a><div className="text-xs text-muted">{when(ev.created_at)}</div></td>
                    <td className="font-mono text-xs">{ev.dataset} · {ev.split}</td>
                    <td className="num">{ev.n}</td>
                    <td className="num text-xs">{Object.entries(ev.metrics).map(([k, v]) => `${k} ${k === "latency" ? v.toFixed(2) + " s" : k === "llm_judge" ? v.toFixed(2) : (v * 100).toFixed(1) + "%"}`).join(" · ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
      {fineTune && <DeleteFineTune runId={String(e.lineage.run_id)} onDeleted={() => { location.hash = "#/models"; }} />}
      <details><summary className="cursor-pointer text-sm text-muted">Raw entry</summary><Code>{JSON.stringify(e, null, 2)}</Code></details>
    </div>
  );
}
