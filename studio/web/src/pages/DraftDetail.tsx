import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, type Draft, type Item, type Mapping, type SplitSpec, type Task } from "../api";
import { Badge, Button, Card, Code, ErrorNote, Field, inputCls, Pager, Tabs, when } from "../ui";

const SPLITS = ["train", "validation", "test"] as const;

/** Guess a column for a task field from names like question/query/sql/db_id. */
function guess(field: string, columns: string[]): string {
  const alias: Record<string, string[]> = {
    question: ["question", "nl", "utterance", "prompt", "input", "text"], db_id: ["db_id", "db", "database", "db_name"],
    sql: ["sql", "query", "gold", "gold_sql", "target"], answer: ["answer", "response", "output", "target", "completion"],
  };
  const cands = [field, ...(alias[field] ?? [])];
  return columns.find((c) => cands.includes(c.toLowerCase())) ?? "";
}

export function DraftDetail({ id }: { id: string }) {
  const qc = useQueryClient();
  const draft = useQuery({
    queryKey: ["draft", id], queryFn: () => api.draft(id),
    refetchInterval: (q) => (q.state.data?.status === "freezing" ? 2000 : false),
  });
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
  const d = draft.data;
  const task = tasks.data?.find((t) => t.ref === (d?.task ?? d?.mapping?.task));
  const refresh = () => { qc.invalidateQueries({ queryKey: ["draft", id] }); qc.invalidateQueries({ queryKey: ["draft-items", id] }); qc.invalidateQueries({ queryKey: ["preview", id] }); };
  const remove = useMutation({ mutationFn: () => api.deleteDraft(id), onSuccess: () => { location.hash = "#/datasets"; } });

  if (!d) return <Card><ErrorNote error={draft.error} />{!draft.error && <p className="text-sm text-muted">Loading…</p>}</Card>;
  return (
    <div className="grid gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <a href="#/datasets" className="text-sm text-muted">← Datasets</a>
          <h1 className="text-2xl font-bold">{d.name}</h1>
          <p className="text-sm text-muted">{d.filename} · {d.n_rows.toLocaleString()} rows · uploaded {when(d.created_at)}</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge value={d.status} />
          {d.frozen_as && <a className="font-mono text-sm text-accent" href={`#/datasets/${d.frozen_as.replace("@", "/")}`}>{d.frozen_as}</a>}
          <Button tone="danger" onClick={() => confirm("Delete this draft? Frozen datasets are not affected.") && remove.mutate()}>Delete draft</Button>
        </div>
      </div>

      <MappingCard d={d} tasks={tasks.data ?? []} onSaved={refresh} />
      {d.mapping && task?.attachments && <AttachmentsCard d={d} task={task} onSaved={refresh} />}
      {d.mapping && <PreviewCard id={id} />}
      {d.mapping && task && <ItemsCard d={d} task={task} onChanged={refresh} />}
      {d.mapping && <FreezeCard d={d} onStarted={refresh} />}
    </div>
  );
}

function MappingCard({ d, tasks, onSaved }: { d: Draft; tasks: Task[]; onSaved: () => void }) {
  const [taskRef, setTaskRef] = useState(d.mapping?.task ?? "");
  const task = tasks.find((t) => t.ref === taskRef);
  const [fields, setFields] = useState<Record<string, string>>(d.mapping?.fields ?? {});
  const [idCol, setIdCol] = useState(d.mapping?.id_column ?? (d.columns.includes("id") ? "id" : ""));
  const [meta, setMeta] = useState<string[]>(d.mapping?.meta_columns ?? []);
  const [split, setSplit] = useState<SplitSpec>(d.mapping?.split ?? { mode: "random", test_frac: 0.1, validation_frac: 0.05, seed: 13 });
  useEffect(() => {  // pick sensible columns when a task is chosen
    if (!task || d.mapping?.task === task.ref) return;
    setFields(Object.fromEntries([...task.inputs, task.target].map((f) => [f, guess(f, d.columns)])));
    if (d.columns.includes("split")) setSplit((s) => ({ ...s, mode: "column", column: "split" }));
  }, [task?.ref]);
  const values = useQuery({ queryKey: ["colvals", d.id, split.column], queryFn: () => api.columnValues(d.id, split.column!), enabled: split.mode === "column" && !!split.column });
  useEffect(() => {  // default label mapping: train/dev/test-looking labels
    if (!values.data || Object.keys(split.values ?? {}).length) return;
    const g = (v: string) => /train/i.test(v) ? "train" : /val/i.test(v) ? "validation" : /test|dev|eval/i.test(v) ? "test" : "";
    setSplit((s) => ({ ...s, values: Object.fromEntries(Object.keys(values.data!.values).map((v) => [v, g(v)])) }));
  }, [values.data]);
  const save = useMutation({
    mutationFn: () => api.setMapping(d.id, { task: taskRef, fields: Object.fromEntries(Object.entries(fields).filter(([, c]) => c)), id_column: idCol || null, meta_columns: meta, split } as Mapping),
    onSuccess: onSaved,
  });
  const optional = task ? Object.keys(task.context) : [];
  const col = (value: string, set: (v: string) => void, allowNone = true) => (
    <select className={inputCls} value={value} onChange={(e) => set(e.target.value)}>
      {allowNone && <option value="">—</option>}
      {d.columns.map((c) => <option key={c} value={c}>{c}</option>)}
    </select>
  );

  return (
    <Card title="1 · Task and columns" actions={d.mapping && <span className="num text-sm text-muted">{SPLITS.map((s) => `${s} ${d.counts[s] ?? 0}`).join(" · ")}{d.dropped_unlabelled ? ` · ${d.dropped_unlabelled} unlabelled dropped` : ""}</span>}>
      <div className="grid gap-4 lg:grid-cols-2">
        <div className="grid content-start gap-3">
          <Field label="Task">
            <select className={inputCls} value={taskRef} onChange={(e) => setTaskRef(e.target.value)}>
              <option value="">Choose a task…</option>
              {tasks.map((t) => <option key={t.ref} value={t.ref}>{t.name} ({t.ref})</option>)}
            </select>
          </Field>
          {task && <p className="text-sm text-muted">{task.description}</p>}
          {task && (
            <div className="grid gap-2 rounded-md border border-line p-3">
              {[...task.inputs, task.target].map((f) => (
                <Field key={f} label={`${f}${f === task.target ? " (target)" : ""}`}>{col(fields[f] ?? "", (v) => setFields({ ...fields, [f]: v }), false)}</Field>
              ))}
              {optional.map((f) => (
                <Field key={f} label={`${f} (optional: rendered from attachments if not mapped)`}>{col(fields[f] ?? "", (v) => setFields({ ...fields, [f]: v }))}</Field>
              ))}
              <Field label="Item id column (optional)">{col(idCol, setIdCol)}</Field>
            </div>
          )}
          {task && (
            <Field label="Extra columns to keep as metadata">
              <div className="flex flex-wrap gap-3">
                {d.columns.filter((c) => !Object.values(fields).includes(c) && c !== idCol && c !== split.column).map((c) => (
                  <label key={c} className="flex items-center gap-1 text-sm">
                    <input type="checkbox" checked={meta.includes(c)} onChange={(e) => setMeta(e.target.checked ? [...meta, c] : meta.filter((x) => x !== c))} />{c}
                  </label>
                ))}
              </div>
            </Field>
          )}
        </div>
        {task && (
          <div className="grid content-start gap-3">
            <Field label="Splits">
              <Tabs value={split.mode} options={[["column", "From a column"], ["random", "Random"], ["single", "All in one split"]]}
                    onChange={(mode) => setSplit({ ...split, mode })} />
            </Field>
            {split.mode === "column" && (
              <>
                <Field label="Split column">{col(split.column ?? "", (v) => setSplit({ ...split, column: v, values: {} }), false)}</Field>
                {values.data && (
                  <div className="grid gap-1">
                    {Object.entries(values.data.values).map(([v, n]) => (
                      <div key={v} className="flex items-center gap-2 text-sm">
                        <span className="w-40 truncate font-mono">{v}</span><span className="num w-16 text-muted">{n}</span>
                        <select className={inputCls} value={split.values?.[v] ?? ""} onChange={(e) => setSplit({ ...split, values: { ...split.values, [v]: e.target.value } })}>
                          <option value="">drop</option>{SPLITS.map((s) => <option key={s} value={s}>{s}</option>)}
                        </select>
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
            {split.mode === "random" && (
              <div className="flex flex-wrap gap-3">
                <Field label="Test %"><input type="number" min={0} max={90} className={`${inputCls} w-24`} value={Math.round((split.test_frac ?? 0) * 100)} onChange={(e) => setSplit({ ...split, test_frac: +e.target.value / 100 })} /></Field>
                <Field label="Validation %"><input type="number" min={0} max={90} className={`${inputCls} w-24`} value={Math.round((split.validation_frac ?? 0) * 100)} onChange={(e) => setSplit({ ...split, validation_frac: +e.target.value / 100 })} /></Field>
              </div>
            )}
            {split.mode === "single" && (
              <Field label="Split"><select className={inputCls} value={split.single ?? "test"} onChange={(e) => setSplit({ ...split, single: e.target.value })}>{SPLITS.map((s) => <option key={s}>{s}</option>)}</select></Field>
            )}
            <div className="flex flex-wrap gap-3">
              <Field label="Move N train items to validation"><input type="number" min={0} className={`${inputCls} w-28`} value={split.validation_n ?? 0} onChange={(e) => setSplit({ ...split, validation_n: +e.target.value })} /></Field>
              <Field label="Keep groups together by" hint="e.g. db_id: a database never appears in two splits">{col(split.group_by ?? "", (v) => setSplit({ ...split, group_by: v || null }))}</Field>
              <Field label="Seed"><input type="number" className={`${inputCls} w-20`} value={split.seed ?? 13} onChange={(e) => setSplit({ ...split, seed: +e.target.value })} /></Field>
            </div>
          </div>
        )}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-3">
        <Button disabled={!task || save.isPending} onClick={() => save.mutate()}>{d.mapping ? "Re-apply mapping" : "Apply mapping"}</Button>
        {d.mapping && (d.edits ?? 0) > 0 && <span className="text-sm text-warn">Re-applying discards {d.edits} item edit{d.edits === 1 ? "" : "s"}.</span>}
      </div>
      <ErrorNote error={save.error} />
      {!d.mapping && d.sample?.length > 0 && (
        <details className="mt-4"><summary className="cursor-pointer text-sm text-muted">First rows of the file</summary>
          <Code>{d.sample.map((r) => JSON.stringify(r)).join("\n")}</Code></details>
      )}
    </Card>
  );
}

function AttachmentsCard({ d, task, onSaved }: { d: Draft; task: Task; onSaved: () => void }) {
  const [path, setPath] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const byPath = useMutation({ mutationFn: () => api.attachPath(d.id, path), onSuccess: onSaved });
  const byZip = useMutation({ mutationFn: () => api.attachZip(d.id, file!), onSuccess: onSaved });
  const a = d.attachments;
  const ok = a && a.n_missing === 0 && (a.needed ?? 0) > 0;
  return (
    <Card title="2 · Attachments" actions={a && <Badge value={ok ? "ok" : "failed"} label={ok ? `all ${a.needed} present` : `${a.n_missing} of ${a.needed} missing`} />}>
      <p className="mb-3 text-sm text-muted">
        {task.name} needs one file per <code className="font-mono">{task.attachments!.key}</code>, laid out as <code className="font-mono">{task.attachments!.layout}</code>.
        Upload a zip (one top folder is stripped), or link a folder on this machine for large sets (no copy is made).
      </p>
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Folder on this machine"><input className={`${inputCls} w-96`} placeholder="/home/you/spider_data/database" value={path} onChange={(e) => setPath(e.target.value)} /></Field>
        <Button tone="plain" disabled={!path || byPath.isPending} onClick={() => byPath.mutate()}>{byPath.isPending ? "Linking…" : "Link folder"}</Button>
        <span className="text-sm text-muted">or</span>
        <Field label="Zip"><input type="file" accept=".zip" className="text-sm" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
        <Button tone="plain" disabled={!file || byZip.isPending} onClick={() => byZip.mutate()}>{byZip.isPending ? "Uploading…" : "Upload zip"}</Button>
      </div>
      <ErrorNote error={byPath.error ?? byZip.error} />
      {a && (a.missing?.length ?? 0) > 0 && <p className="mt-3 font-mono text-xs text-warn">missing: {a.missing!.join(", ")}{(a.n_missing ?? 0) > a.missing!.length ? " …" : ""}</p>}
      {a && <p className="mt-2 text-xs text-muted">{a.files} files attached</p>}
    </Card>
  );
}

function PreviewCard({ id }: { id: string }) {
  const preview = useQuery({ queryKey: ["preview", id], queryFn: () => api.preview(id, 3) });
  return (
    <Card title="3 · Prompt preview" actions={<span className="text-sm text-muted">exactly what training, evaluation and the playground send</span>}>
      <ErrorNote error={preview.error} />
      <div className="grid gap-4">
        {(preview.data ?? []).map((p) => (
          <div key={p.id} className="rounded-md border border-line p-3">
            <div className="mb-2 flex gap-3 font-mono text-xs text-muted"><span>{p.id}</span><span>{p.split}</span>{p.est_tokens && <span>~{p.est_tokens} tokens</span>}</div>
            {p.error ? <p className="text-sm text-warn">{p.error}</p> : (
              <div className="grid gap-2">
                {p.messages!.map((m, i) => (
                  <div key={i}><div className="font-mono text-xs uppercase text-muted">{m.role}</div><Code className="max-h-64 overflow-y-auto">{m.content}</Code></div>
                ))}
                <div><div className="font-mono text-xs uppercase text-muted">target</div><Code>{p.target}</Code></div>
              </div>
            )}
          </div>
        ))}
      </div>
    </Card>
  );
}

function ItemsCard({ d, task, onChanged }: { d: Draft; task: Task; onChanged: () => void }) {
  const [split, setSplit] = useState<string>("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const [editing, setEditing] = useState<Item | null>(null);
  const items = useQuery({ queryKey: ["draft-items", d.id, split, q, offset], queryFn: () => api.draftItems(d.id, { split, q, offset, limit: 15 }) });
  const fields = useMemo(() => [...task.inputs, task.target], [task]);
  return (
    <Card title="4 · Items" actions={<Tabs value={split} options={[["", "all"], ...SPLITS.map((s) => [s, `${s} ${d.counts[s] ?? 0}`] as [string, string])]} onChange={(s) => { setSplit(s); setOffset(0); }} />}>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <input className={`${inputCls} w-72`} placeholder="Search any field" value={q} onChange={(e) => { setQ(e.target.value); setOffset(0); }} />
        <Pager total={items.data?.total ?? 0} offset={offset} limit={15} onChange={setOffset} />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted"><tr><th className="py-2 pr-2">id</th><th className="pr-2">split</th>{fields.map((f) => <th key={f} className="pr-2">{f}</th>)}<th /></tr></thead>
          <tbody>
            {(items.data?.items ?? []).map((it) => (
              <tr key={it.id} className="border-t border-line align-top">
                <td className="whitespace-nowrap py-2 pr-2 font-mono text-xs">{it.id}{it.edited ? <span className="ml-1 text-mango">●</span> : null}</td>
                <td className="pr-2 font-mono text-xs">{it.split}</td>
                {fields.map((f) => <td key={f} dir="auto" className={`max-w-md pr-2 ${f === task.target ? "font-mono text-xs" : ""}`}>{String(it[f] ?? "")}</td>)}
                <td><Button tone="plain" onClick={() => setEditing(it)}>Edit</Button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {editing && <ItemEditor draftId={d.id} item={editing} fields={fields} onClose={() => setEditing(null)} onChanged={() => { onChanged(); items.refetch(); }} />}
    </Card>
  );
}

function ItemEditor({ draftId, item, fields, onClose, onChanged }: { draftId: string; item: Item; fields: string[]; onClose: () => void; onChanged: () => void }) {
  const [vals, setVals] = useState<Record<string, string>>(Object.fromEntries([...fields, "split"].map((f) => [f, String(item[f] ?? "")])));
  const save = useMutation({ mutationFn: () => api.editItem(draftId, item.id, vals), onSuccess: () => { onChanged(); onClose(); } });
  const del = useMutation({ mutationFn: () => api.deleteItem(draftId, item.id), onSuccess: () => { onChanged(); onClose(); } });
  return (
    <div className="mt-4 rounded-md border border-accent p-4">
      <div className="mb-3 flex items-center justify-between"><h3 className="font-bold">Edit <span className="font-mono">{item.id}</span></h3><Button tone="plain" onClick={onClose}>Close</Button></div>
      <div className="grid gap-3">
        <Field label="split"><select className={`${inputCls} w-40`} value={vals.split} onChange={(e) => setVals({ ...vals, split: e.target.value })}>{SPLITS.map((s) => <option key={s}>{s}</option>)}</select></Field>
        {fields.map((f) => (
          <Field key={f} label={f}><textarea dir="auto" rows={Math.min(8, Math.max(2, Math.ceil(vals[f].length / 90)))} className={`${inputCls} font-mono`} value={vals[f]} onChange={(e) => setVals({ ...vals, [f]: e.target.value })} /></Field>
        ))}
      </div>
      <div className="mt-3 flex gap-3">
        <Button disabled={save.isPending} onClick={() => save.mutate()}>Save</Button>
        <Button tone="danger" disabled={del.isPending} onClick={() => del.mutate()}>Delete item</Button>
      </div>
      <ErrorNote error={save.error ?? del.error} />
    </div>
  );
}

function FreezeCard({ d, onStarted }: { d: Draft; onStarted: () => void }) {
  const [dsId, setDsId] = useState(d.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").replace(/-\d+-\d+$/, "") || "dataset");
  const [version, setVersion] = useState("v1.0");
  const [dropInvalid, setDropInvalid] = useState(false);
  const [checkGold, setCheckGold] = useState(true);
  const go = useMutation({ mutationFn: () => api.freeze(d.id, { id: dsId, version, drop_invalid: dropInvalid, check_gold: checkGold }), onSuccess: onStarted });
  const v = d.validation;
  return (
    <Card title="5 · Freeze a version" actions={<Badge value={d.status} />}>
      <p className="mb-3 text-sm text-muted">Freezing checks every item (required fields, attachments, and for SQL tasks that each gold query runs), renders the context into the rows, and writes an immutable version.</p>
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Dataset id"><input className={inputCls} value={dsId} onChange={(e) => setDsId(e.target.value)} /></Field>
        <Field label="Version"><input className={`${inputCls} w-24`} value={version} onChange={(e) => setVersion(e.target.value)} /></Field>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={checkGold} onChange={(e) => setCheckGold(e.target.checked)} />Run every gold query</label>
        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={dropInvalid} onChange={(e) => setDropInvalid(e.target.checked)} />Drop invalid items instead of failing</label>
        <Button disabled={go.isPending || d.status === "freezing" || d.status === "frozen"} onClick={() => go.mutate()}>{d.status === "freezing" ? "Freezing…" : "Freeze"}</Button>
      </div>
      <ErrorNote error={go.error ?? d.error} />
      {d.status === "freezing" && d.progress && <p className="mt-2 num text-sm text-muted">checked {d.progress.checked} of {d.progress.total}</p>}
      {v && (
        <div className="mt-3 text-sm">
          <p>{v.checked} checked, <b className={v.invalid ? "text-warn" : ""}>{v.invalid} invalid</b>{Object.keys(v.reasons).length ? `: ${Object.entries(v.reasons).map(([k, n]) => `${k} (${n})`).join(", ")}` : ""}</p>
          {v.examples.length > 0 && <p className="mt-1 font-mono text-xs text-muted">{v.examples.map((e) => e.id).join(", ")} — search for them in Items to fix or delete</p>}
        </div>
      )}
      {d.frozen_as && <p className="mt-3 text-sm">Frozen as <a className="font-mono text-accent" href={`#/datasets/${d.frozen_as.replace("@", "/")}`}>{d.frozen_as}</a>.</p>}
    </Card>
  );
}
