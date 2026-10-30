import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api, type ColumnProfile, type DatasetProfile, type DatasetRef, type Item } from "../api";
import { BarChart } from "../BarChart";
import { Badge, Button, Card, Code, ErrorNote, Field, inputCls, Pager, Tabs, when } from "../ui";

const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (i: number) => css(`--color-series-${Math.min(i, 7) + 1}`);
const bytes = (n: number) => n < 1024 ? `${n} B` : n < 1024 ** 2 ? `${(n / 1024).toFixed(1)} KB` : n < 1024 ** 3 ? `${(n / 1024 ** 2).toFixed(1)} MB` : `${(n / 1024 ** 3).toFixed(2)} GB`;
const str = (v: unknown) => v === null || v === undefined ? "" : typeof v === "object" ? JSON.stringify(v) : String(v);

type Tab = "overview" | "data" | "columns" | "databases" | "task" | "usage";

/** Everything about one frozen dataset version: metadata, profile, every item, its databases, task and usage. */
export function DatasetExplorer({ id, version, query: params }: { id: string; version: string; query: URLSearchParams }) {
  const [tab, setTabState] = useState<Tab>((params.get("tab") as Tab) || "overview");
  const [dbKey, setDbKey] = useState(params.get("db") ?? "");
  const [filter, setFilter] = useState<{ field: string; value: string } | null>(null);
  const setTab = (t: Tab, extra = "") => { setTabState(t); history.replaceState(null, "", `#/datasets/${id}/${version}?tab=${t}${extra}`); };
  const qs = params.toString();
  useEffect(() => { setTabState((params.get("tab") as Tab) || "overview"); setDbKey(params.get("db") ?? ""); }, [id, version, qs]);  // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => setFilter(null), [id, version]);

  const ref = useQuery({ queryKey: ["dataset", id, version], queryFn: () => api.dataset(id, version) });
  const prof = useQuery({ queryKey: ["ds-profile", id, version], queryFn: () => api.datasetProfile(id, version) });
  const all = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const copy = useMutation({ mutationFn: () => api.draftFromDataset(ref.data!.path), onSuccess: (d) => { location.hash = `#/drafts/${d.id}`; } });
  const p = prof.data;
  const r = ref.data;

  const tabs: [Tab, string][] = [["overview", "Overview"], ["data", "Data"], ["columns", "Columns"],
    ...(p?.attachment_key ? [["databases", "Databases"] as [Tab, string]] : []), ["task", "Task"], ["usage", "Usage"]];
  const showItemsOf = (field: string, value: string) => { setFilter({ field, value }); setTab("data"); };

  return (
    <div className="grid gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div className="min-w-0">
          <a href="#/datasets" className="text-sm text-muted">← Datasets</a>
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="font-mono text-2xl font-bold">{id}@{version}</h1>
            <select aria-label="Switch dataset" className={inputCls} value={`${id}/${version}`} onChange={(e) => { location.hash = `#/datasets/${e.target.value}`; }}>
              {(all.data ?? []).map((d) => <option key={d.path} value={`${d.id}/${d.version}`}>{d.id}@{d.version}</option>)}
            </select>
          </div>
          <p className="text-sm text-muted">{p?.task.name ?? r?.task} · {r ? Object.values(r.counts).reduce((a, b) => a + b, 0).toLocaleString() : "…"} items · created {when(r?.created_at)}</p>
        </div>
        <div className="flex flex-wrap gap-2">
          <a className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-accent-soft" href={`#/training?dataset=${encodeURIComponent(r?.path ?? "")}`}>Fine-tune on it</a>
          <a className="rounded-md border border-line px-3 py-1.5 text-sm hover:bg-accent-soft" href={`#/evals/new?dataset=${encodeURIComponent(r?.path ?? "")}`}>Evaluate models on it</a>
          {r?.task && <Button tone="plain" disabled={copy.isPending} onClick={() => copy.mutate()}>Edit as a new version</Button>}
        </div>
      </div>
      <ErrorNote error={ref.error ?? prof.error ?? copy.error} />
      <Tabs value={tab} options={tabs} onChange={(t) => setTab(t)} />
      {r && p ? <>
        {tab === "overview" && <Overview r={r} p={p} openDb={(k) => { setDbKey(k); setTab("databases", `&db=${encodeURIComponent(k)}`); }} />}
        {tab === "data" && <DataTab id={id} version={version} r={r} p={p} filter={filter} setFilter={setFilter} />}
        {tab === "columns" && <ColumnsTab p={p} showItemsOf={showItemsOf} />}
        {tab === "databases" && <DatabasesTab id={id} version={version} p={p} dbKey={dbKey}
          setDbKey={(k) => { setDbKey(k); setTab("databases", `&db=${encodeURIComponent(k)}`); }} showItemsOf={showItemsOf} />}
        {tab === "task" && <TaskTab p={p} />}
        {tab === "usage" && <UsageTab tag={`${id}@${version}`} />}
      </> : !ref.error && <p className="text-sm text-muted">Profiling the dataset…</p>}
    </div>
  );
}

function Dl({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <dl className="grid gap-x-6 gap-y-2 text-sm sm:grid-cols-[10rem_1fr]">
      {rows.filter(([, v]) => v !== undefined && v !== null && v !== "").map(([k, v]) => (
        <div key={k} className="contents"><dt className="font-mono text-xs uppercase text-muted sm:pt-0.5">{k}</dt><dd className="min-w-0 break-words">{v}</dd></div>
      ))}
    </dl>
  );
}

function Overview({ r, p, openDb }: { r: DatasetRef; p: DatasetProfile; openDb: (k: string) => void }) {
  const splits = Object.keys(r.counts);
  const hist = p.tokens_hist;
  const cov = p.coverage;
  const src = r.source as { draft?: string; filename?: string; rows?: number; edits?: number } | undefined;
  return (
    <div className="grid gap-6">
      <dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {splits.map((s) => (
          <div key={s} className="rounded-lg border border-line bg-surface p-4">
            <dt className="font-mono text-xs uppercase text-muted">{s}</dt>
            <dd className="num text-3xl">{r.counts[s].toLocaleString()}</dd>
            <dd className="text-xs text-muted">
              {p.tokens[s] && <>~<span className="num">{p.tokens[s].p50}</span> tokens median, <span className="num">{p.tokens[s].max}</span> max</>}
              {cov?.per_split[s] && <> · <span className="num">{cov.per_split[s].keys}</span> {cov.key === "db_id" ? "databases" : cov.key}</>}
            </dd>
          </div>
        ))}
      </dl>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Metadata">
          <Dl rows={[
            ["id", <span className="font-mono">{r.id}</span>], ["version", <span className="font-mono">{r.version}</span>],
            ["task", <span className="font-mono">{p.task.ref}</span>], ["created", when(r.created_at)],
            ["inputs", <span className="font-mono">{p.fields.inputs.join(", ")}</span>],
            ["target", <span className="font-mono">{p.fields.target}</span>],
            ["context", p.fields.context.length ? <span className="font-mono">{p.fields.context.join(", ")} <span className="text-muted">(rendered from the attachments)</span></span> : null],
            ["split by", <span className="font-mono text-xs">{r.split_by}</span>],
            ["source", src ? <span>{src.filename} · {src.rows?.toLocaleString()} rows{src.edits ? ` · ${src.edits} edited` : ""} · <a className="text-accent" href={`#/drafts/${src.draft}`}>draft</a></span> : null],
            ["validation", r.validation ? `${r.validation.checked.toLocaleString()} checked at freeze, ${r.validation.invalid} invalid dropped` : null],
            ["sha256", <span className="break-all font-mono text-xs">{r.sha256}</span>],
          ]} />
        </Card>
        <Card title="Files">
          <table className="w-full text-sm">
            <tbody>
              {p.files.map((f) => {
                const split = f.name.endsWith(".parquet") ? f.name.replace(".parquet", "") : null;
                return (
                  <tr key={f.name} className="border-t border-line first:border-t-0">
                    <td className="py-1.5 font-mono text-xs">{f.name}</td>
                    <td className="num text-right text-xs text-muted">{bytes(f.bytes)}</td>
                    <td className="text-right text-xs">{split && split in r.counts && (["jsonl", "csv", "parquet"] as const).map((fmt) => (
                      <a key={fmt} className="ml-2 text-accent" href={api.downloadUrl(r.id, r.version, split, fmt)} download>{fmt}</a>))}</td>
                  </tr>
                );
              })}
              {p.attachments && (
                <tr className="border-t border-line"><td className="py-1.5 font-mono text-xs">attachments/</td>
                  <td className="num text-right text-xs text-muted">{bytes(p.attachments.bytes)}</td>
                  <td className="text-right text-xs text-muted">{p.attachments.files} files</td></tr>
              )}
            </tbody>
          </table>
          <p className="mt-2 font-mono text-xs text-muted break-all">{r.path}</p>
        </Card>
      </div>

      {hist && (
        <Card title="Prompt size per item">
          <BarChart categories={hist.categories} format={(v) => `${v.toFixed(1)}%`}
            series={splits.map((s, i) => ({ name: s, color: color(i), values: hist.series[s].map((n) => r.counts[s] ? (100 * n) / r.counts[s] : 0) }))} />
          <p className="text-xs text-muted">Estimated tokens (characters ÷ 3.2) of everything a model reads and writes for an item: inputs, context and target. Share of each split, so splits of different sizes compare.</p>
        </Card>
      )}

      {cov && (
        <Card title={cov.key === "db_id" ? "Databases per split" : `${cov.key} per split`}>
          <div className="grid gap-4 md:grid-cols-3">
            {splits.map((s) => (
              <div key={s} className="min-w-0">
                <div className="mb-1 font-mono text-xs uppercase text-muted">{s} · {cov.per_split[s]?.keys} {cov.key === "db_id" ? "databases" : "keys"}</div>
                <ul className="grid gap-0.5 text-sm">
                  {(cov.per_split[s]?.top ?? []).map(([k, n]) => (
                    <li key={k} className="flex justify-between gap-2"><button className="truncate text-left text-accent" onClick={() => openDb(k)}>{k}</button><span className="num text-muted">{n}</span></li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <p className="mt-3 text-sm text-muted">
            Shared between splits: {Object.entries(cov.shared).map(([k, n]) => `${k.replace("|", " & ")} ${n}`).join(" · ")}.
            {Object.values(cov.shared).every((n) => n === 0) && " No database appears in two splits, so validation and test measure how a model handles schemas it never trained on."}
          </p>
        </Card>
      )}
    </div>
  );
}

function DataTab({ id, version, r, p, filter, setFilter }: {
  id: string; version: string; r: DatasetRef; p: DatasetProfile;
  filter: { field: string; value: string } | null; setFilter: (f: { field: string; value: string } | null) => void;
}) {
  const splits = Object.keys(r.counts);
  const [split, setSplit] = useState(splits[0]);
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(25);
  const [open, setOpen] = useState<string | null>(null);
  const [view, setView] = useState<"table" | "cards">("table");
  useEffect(() => { setOffset(0); setOpen(null); }, [split, q, filter, limit]);
  // a filter set from another tab: jump to the first split that has matching items
  const firstWith = useQuery({
    queryKey: ["ds-filter-split", id, version, filter], enabled: !!filter,
    queryFn: async () => {
      for (const s of splits) if ((await api.datasetItems(id, version, { split: s, limit: 1, ...filter! })).total) return s;
      return splits[0];
    },
  });
  useEffect(() => { if (firstWith.data) setSplit(firstWith.data); }, [firstWith.data]);

  const items = useQuery({ queryKey: ["ds-items", id, version, split, q, offset, limit, filter],
    queryFn: () => api.datasetItems(id, version, { split, q, offset, limit, ...(filter ?? {}) }) });
  const cols = ["id", ...p.fields.inputs, p.fields.target];
  const metaCols = p.columns.filter((c) => c.role === "metadata" || c.role === "other").map((c) => c.name).filter((c) => !c.includes(".")).slice(0, 3);

  return (
    <Card title="Items" actions={<Tabs value={split} options={splits.map((s) => [s, `${s} ${r.counts[s]}`])} onChange={setSplit} />}>
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <Field label="Search all fields"><input className={`${inputCls} w-64`} placeholder="text, SQL, id…" value={q} onChange={(e) => setQ(e.target.value)} /></Field>
        {filter && <span className="flex items-center gap-2 rounded-md bg-accent-soft px-2 py-1 text-sm"><span className="font-mono text-xs">{filter.field} = {filter.value}</span>
          <button aria-label="Clear filter" className="text-muted" onClick={() => setFilter(null)}>✕</button></span>}
        <Field label="Per page"><select className={inputCls} value={limit} onChange={(e) => setLimit(+e.target.value)}>{[25, 50, 100, 500].map((n) => <option key={n}>{n}</option>)}</select></Field>
        <Tabs value={view} options={[["table", "Table"], ["cards", "Cards"]]} onChange={setView} />
        <div className="ml-auto"><Pager total={items.data?.total ?? 0} offset={offset} limit={limit} onChange={setOffset} /></div>
      </div>
      <ErrorNote error={items.error} />
      {view === "table" ? (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead className="text-left font-mono text-xs uppercase text-muted"><tr>{[...cols, ...metaCols].map((c) => <th key={c} className="py-2 pr-3">{c}</th>)}</tr></thead>
            <tbody>
              {(items.data?.items ?? []).map((it) => (
                <ItemRows key={it.id} id={id} version={version} split={split} it={it} cols={cols} metaCols={metaCols} target={p.fields.target}
                  open={open === it.id} toggle={() => setOpen(open === it.id ? null : it.id)} />
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="grid gap-2">
          {(items.data?.items ?? []).map((it) => (
            <div key={it.id} className="rounded-md border border-line p-3 text-sm">
              <button className="mb-1 flex w-full gap-2 text-left font-mono text-xs text-muted" onClick={() => setOpen(open === it.id ? null : it.id)} aria-expanded={open === it.id}>
                <span>{it.id}</span>{it.edited ? <Badge value="mapped" label="edited" /> : null}<span className="ml-auto">{open === it.id ? "less" : "everything"}</span>
              </button>
              {[...p.fields.inputs, p.fields.target].map((f) => (
                <div key={f} className="grid grid-cols-[6rem_1fr] gap-2">
                  <span className="font-mono text-xs uppercase text-muted">{f}</span>
                  <span dir="auto" className={f === p.fields.target ? "font-mono text-xs" : ""}>{str(it[f])}</span>
                </div>
              ))}
              {open === it.id && <ItemDetail id={id} version={version} split={split} itemId={it.id} />}
            </div>
          ))}
        </div>
      )}
      {items.data?.total === 0 && <p className="text-sm text-muted">No items match.</p>}
    </Card>
  );
}

function ItemRows({ id, version, split, it, cols, metaCols, target, open, toggle }: {
  id: string; version: string; split: string; it: Item; cols: string[]; metaCols: string[]; target: string; open: boolean; toggle: () => void;
}) {
  return (
    <>
      <tr className={`cursor-pointer border-t border-line align-top hover:bg-accent-soft ${open ? "bg-accent-soft" : ""}`} onClick={toggle} aria-expanded={open}>
        {[...cols, ...metaCols].map((c) => (
          <td key={c} dir="auto" className={`max-w-80 py-1.5 pr-3 ${c === "id" || c === target ? "font-mono text-xs" : ""}`}>
            <span className="line-clamp-2 break-words">{str(it[c])}</span>
          </td>
        ))}
      </tr>
      {open && <tr><td colSpan={cols.length + metaCols.length} className="pb-3"><ItemDetail id={id} version={version} split={split} itemId={it.id} /></td></tr>}
    </>
  );
}

function ItemDetail({ id, version, split, itemId }: { id: string; version: string; split: string; itemId: string }) {
  const d = useQuery({ queryKey: ["ds-item", id, version, split, itemId], queryFn: () => api.datasetItem(id, version, split, itemId) });
  const [view, setView] = useState<"fields" | "prompt" | "json">("fields");
  if (d.error) return <ErrorNote error={d.error} />;
  if (!d.data) return <p className="text-xs text-muted">Loading…</p>;
  const { row, messages } = d.data;
  return (
    <div className="mt-2 grid gap-2 rounded-md border border-line p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Tabs value={view} options={[["fields", "All fields"], ["prompt", "Prompt the model gets"], ["json", "Raw JSON"]]} onChange={setView} />
        <span className="text-xs text-muted">~<span className="num">{d.data.est_prompt_tokens.toLocaleString()}</span> prompt tokens{!d.data.applies && " · not used by this task (filtered out)"}</span>
      </div>
      {view === "fields" && (
        <div className="grid gap-2">
          {Object.entries(row).map(([k, v]) => {
            const long = typeof v === "string" && v.length > 300;
            let shown = v;
            if (k === "meta" && typeof v === "string") { try { shown = JSON.parse(v); } catch { /* keep the string */ } }
            return (
              <div key={k} className="grid gap-1 sm:grid-cols-[8rem_1fr]">
                <span className="font-mono text-xs uppercase text-muted">{k}</span>
                {long ? <details><summary className="cursor-pointer text-xs text-muted">{(v as string).length.toLocaleString()} chars</summary><Code className="max-h-96 overflow-y-auto">{v as string}</Code></details>
                  : typeof shown === "object" && shown !== null ? <Code>{JSON.stringify(shown, null, 2)}</Code>
                  : <span dir="auto" className="break-words text-sm">{str(shown)}</span>}
              </div>
            );
          })}
        </div>
      )}
      {view === "prompt" && (d.data.error ? <p className="text-xs text-warn">{d.data.error}</p>
        : (messages ?? []).map((m, i) => <div key={i}><div className="font-mono text-xs uppercase text-muted">{m.role}</div><Code className="max-h-96 overflow-y-auto">{m.content}</Code></div>))}
      {view === "json" && <Code className="max-h-96 overflow-y-auto">{JSON.stringify(row, null, 2)}</Code>}
    </div>
  );
}

const ROLE_TONE: Record<string, string> = { input: "mapped", "input (attachment key)": "mapped", target: "succeeded", context: "running", metadata: "queued" };

function ColumnsTab({ p, showItemsOf }: { p: DatasetProfile; showItemsOf: (field: string, value: string) => void }) {
  const n = p.columns[0] ? p.columns[0].nulls + p.columns[0].distinct : 0;
  return (
    <Card title={`Columns (${p.columns.length})`}>
      <p className="mb-3 text-sm text-muted">Every column across all splits. Struct fields and the JSON <code className="font-mono">meta</code> column are expanded one level. Lengths are in characters.</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[900px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted">
            <tr><th className="py-2 pr-3">Column</th><th className="pr-3">Type</th><th className="pr-3">Role</th><th className="pr-3 text-right">Empty</th><th className="pr-3 text-right">Distinct</th><th className="pr-3">Length / range</th><th>Most common</th></tr>
          </thead>
          <tbody>{p.columns.map((c) => <ColumnRow key={c.name} c={c} total={n} showItemsOf={showItemsOf} />)}</tbody>
        </table>
      </div>
    </Card>
  );
}

function ColumnRow({ c, total, showItemsOf }: { c: ColumnProfile; total: number; showItemsOf: (field: string, value: string) => void }) {
  const [more, setMore] = useState(false);
  const top = c.top ?? [];
  const filterable = !c.name.includes(".") && c.role !== "context";
  return (
    <tr className="border-t border-line align-top [&>td]:py-2">
      <td className="pr-3 font-mono text-xs">{c.name}</td>
      <td className="pr-3 font-mono text-xs text-muted">{c.type}</td>
      <td className="pr-3"><Badge value={ROLE_TONE[c.role] ?? "unknown"} label={c.role} /></td>
      <td className="num pr-3 text-right">{c.nulls ? c.nulls.toLocaleString() : "—"}</td>
      <td className="num pr-3 text-right">{c.distinct.toLocaleString()}{c.distinct === total && total > 0 ? <div className="text-xs text-muted">unique</div> : null}</td>
      <td className="num pr-3 text-xs">{c.chars ? `${c.chars.min}–${c.chars.max} · median ${c.chars.p50} · p90 ${c.chars.p90}` : c.numbers ? `${c.numbers.min}–${c.numbers.max} · median ${c.numbers.p50}` : "—"}</td>
      <td className="max-w-96 text-xs">
        {top.length ? <div className="flex flex-wrap gap-1">
          {(more ? top : top.slice(0, 4)).map(([v, k]) => (
            <button key={v} disabled={!filterable} onClick={() => showItemsOf(c.name, v)} title={filterable ? "Show these items" : v}
              className={`max-w-56 truncate rounded border border-line px-1.5 py-0.5 text-left ${filterable ? "hover:bg-accent-soft" : "cursor-default"}`}>
              <span className="font-mono">{v.length > 60 ? `${v.slice(0, 60)}…` : v}</span> <span className="num text-muted">{k}</span>
            </button>
          ))}
          {top.length > 4 && <button className="text-muted" onClick={() => setMore(!more)}>{more ? "less" : `+${top.length - 4}`}</button>}
        </div> : <span className="text-muted">all different</span>}
      </td>
    </tr>
  );
}

function DatabasesTab({ id, version, p, dbKey, setDbKey, showItemsOf }: {
  id: string; version: string; p: DatasetProfile; dbKey: string; setDbKey: (k: string) => void; showItemsOf: (f: string, v: string) => void;
}) {
  const keys = useQuery({ queryKey: ["attachments", id, version], queryFn: () => api.attachments(id, version) });
  const [q, setQ] = useState("");
  const list = (keys.data?.keys ?? []).filter((k) => k.toLowerCase().includes(q.toLowerCase()));
  const split = (k: string) => Object.entries(p.coverage?.per_key[k] ?? {}).map(([s, n]) => `${s} ${n}`).join(", ") || "no items";
  useEffect(() => { if (!dbKey && keys.data?.keys.length) setDbKey(keys.data.keys[0]); }, [keys.data]);  // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <div className="grid gap-6 lg:grid-cols-[16rem_1fr]">
      <Card title={`${keys.data?.keys.length ?? "…"} databases`}>
        <input className={`${inputCls} mb-2 w-full`} placeholder="Filter" value={q} onChange={(e) => setQ(e.target.value)} />
        <ul className="grid max-h-[60vh] gap-0.5 overflow-y-auto text-sm">
          {list.map((k) => (
            <li key={k}><button onClick={() => setDbKey(k)} className={`w-full truncate rounded px-2 py-1 text-left font-mono text-xs ${k === dbKey ? "bg-accent text-bg" : "hover:bg-accent-soft"}`}>
              {k}<span className={`block ${k === dbKey ? "" : "text-muted"}`}>{split(k)}</span></button></li>
          ))}
        </ul>
      </Card>
      {dbKey ? <Database id={id} version={version} dbKey={dbKey} attKey={p.attachment_key!} showItemsOf={showItemsOf} /> : <p className="text-sm text-muted">Pick a database.</p>}
    </div>
  );
}

function Database({ id, version, dbKey, attKey, showItemsOf }: { id: string; version: string; dbKey: string; attKey: string; showItemsOf: (f: string, v: string) => void }) {
  const db = useQuery({ queryKey: ["db", id, version, dbKey], queryFn: () => api.database(id, version, dbKey) });
  const ctx = useQuery({ queryKey: ["context", id, version, dbKey], queryFn: () => api.context(id, version, dbKey) });
  const [table, setTable] = useState<string | null>(null);
  useEffect(() => setTable(null), [dbKey]);
  const d = db.data;
  return (
    <div className="grid min-w-0 gap-6">
      <ErrorNote error={db.error} />
      {d && (
        <Card title={<span className="font-mono">{d.key}</span>} actions={
          <Button tone="plain" onClick={() => showItemsOf(attKey, d.key)}>Show its {Object.values(d.items).reduce((a, b) => a + b, 0)} items</Button>}>
          <p className="mb-3 text-sm text-muted">
            {d.tables.length} tables · {bytes(d.bytes)} · items: {Object.entries(d.items).filter(([, n]) => n).map(([s, n]) => `${s} ${n}`).join(" · ") || "none"}
            <span className="block break-all font-mono text-xs">{d.file}</span>
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            {d.tables.map((t) => (
              <div key={t.name} className={`rounded-md border p-3 ${table === t.name ? "border-accent" : "border-line"}`}>
                <div className="mb-1 flex items-center justify-between gap-2">
                  <button className="font-mono text-sm font-bold text-accent" onClick={() => setTable(table === t.name ? null : t.name)}>{t.name}</button>
                  <span className="num text-xs text-muted">{t.rows?.toLocaleString() ?? "?"} rows</span>
                </div>
                <ul className="grid gap-0.5 font-mono text-xs">
                  {t.columns.map((c) => {
                    const fk = t.foreign_keys.find((f) => f.column === c.name);
                    return <li key={c.name} className="flex flex-wrap gap-x-2"><span className={c.pk ? "font-bold" : ""}>{c.name}</span><span className="text-muted">{c.type || "—"}</span>
                      {c.pk && <span className="text-mango">PK</span>}{fk && <span className="text-accent">→ {fk.ref_table}.{fk.ref_column}</span>}</li>;
                  })}
                </ul>
              </div>
            ))}
          </div>
        </Card>
      )}
      {table && <TableBrowser id={id} version={version} dbKey={dbKey} table={table} />}
      {ctx.data && Object.entries(ctx.data).map(([k, v]) => (
        <Card key={k} title={`${k} as the model sees it`}>
          <p className="mb-2 text-sm text-muted">The context rendered into every prompt for this database ({v.length.toLocaleString()} chars, ~{Math.round(v.length / 3.2).toLocaleString()} tokens).</p>
          <Code className="max-h-[32rem] overflow-y-auto">{v}</Code>
        </Card>
      ))}
    </div>
  );
}

function TableBrowser({ id, version, dbKey, table }: { id: string; version: string; dbKey: string; table: string }) {
  const [offset, setOffset] = useState(0);
  useEffect(() => setOffset(0), [table]);
  const rows = useQuery({ queryKey: ["table", id, version, dbKey, table, offset], queryFn: () => api.tableRows(id, version, dbKey, table, { offset, limit: 50 }) });
  return (
    <Card title={<span className="font-mono">{table}</span>} actions={<Pager total={rows.data?.total ?? 0} offset={offset} limit={50} onChange={setOffset} />}>
      <ErrorNote error={rows.error} />
      <div className="max-h-[32rem] overflow-auto">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-surface text-left font-mono uppercase text-muted"><tr>{rows.data?.columns.map((c) => <th key={c} className="py-1.5 pr-3">{c}</th>)}</tr></thead>
          <tbody>
            {rows.data?.rows.map((r, i) => (
              <tr key={i} className="border-t border-line">{r.map((v, j) => <td key={j} dir="auto" className={`max-w-64 truncate py-1 pr-3 ${typeof v === "number" ? "num" : ""}`}>{v === null ? <span className="text-muted">NULL</span> : String(v)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function TaskTab({ p }: { p: DatasetProfile }) {
  const t = p.task as DatasetProfile["task"] & { decoding?: Record<string, unknown>; rubric?: string | null; row_filter?: Record<string, unknown> };
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card title={t.name}>
        <p className="mb-3 text-sm text-muted">{t.description}</p>
        <Dl rows={[
          ["task", <span className="font-mono">{t.ref}</span>],
          ["inputs", <span className="font-mono">{t.inputs.join(", ")}</span>], ["target", <span className="font-mono">{t.target}</span>],
          ["context", Object.keys(t.context).length ? <span className="font-mono">{Object.entries(t.context).map(([k, v]) => `${k} (${v.type})`).join(", ")}</span> : null],
          ["attachments", t.attachments ? <span className="font-mono text-xs">{t.attachments.type} · {t.attachments.layout}</span> : null],
          ["applies to", t.row_filter && Object.keys(t.row_filter).length ? <span className="font-mono text-xs">{JSON.stringify(t.row_filter)}</span> : null],
          ["decoding", t.decoding ? <span className="font-mono text-xs">{JSON.stringify(t.decoding)}</span> : null],
          ["metrics", <span className="text-xs">{t.metrics.available.map((m) => t.metrics.default.includes(m) ? `${m} (default)` : m).join(" · ")}</span>],
          ["judge rubric", t.rubric ? <span className="font-mono text-xs">{t.rubric}</span> : null],
        ]} />
      </Card>
      <Card title="Prompt template">
        <div className="font-mono text-xs uppercase text-muted">system</div>
        <Code>{t.system || "(none)"}</Code>
        <div className="mt-3 font-mono text-xs uppercase text-muted">user</div>
        <Code>{t.user_template}</Code>
        <p className="mt-2 text-xs text-muted">Training, evaluation and the playground all render items through this template, so a fine-tune learns exactly the prompt it is evaluated on. See any item's "Prompt the model gets" in Data.</p>
      </Card>
    </div>
  );
}

function UsageTab({ tag }: { tag: string }) {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const evals = useQuery({ queryKey: ["evals"], queryFn: api.evals });
  const trained = (models.data ?? []).filter((m) => m.lineage?.dataset === tag);
  const used = (evals.data ?? []).filter((e) => e.dataset === tag);
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <Card title={`Models fine-tuned on it (${trained.length})`}>
        {trained.length ? <ul className="grid gap-2 text-sm">{trained.map((m) => (
          <li key={m.id}><a className="font-mono text-accent" href={`#/models/${m.id}`}>{m.id}</a>
            <div className="text-xs text-muted">on {m.parent ?? m.hf_id}{m.lineage?.train_fraction ? ` · ${Math.round(Number(m.lineage.train_fraction) * 100)}% of train` : ""} · {when(m.created_at)}</div></li>
        ))}</ul> : <p className="text-sm text-muted">None yet. <a className="text-accent" href="#/training">Fine-tune a model</a>.</p>}
      </Card>
      <Card title={`Evaluations on it (${used.length})`}>
        {used.length ? <ul className="grid gap-2 text-sm">{used.map((e) => (
          <li key={e.id}><a className="font-mono text-accent" href={`#/evals/${e.id}`}>{e.id}</a> <Badge value={e.status} />
            <div className="text-xs text-muted">{(e.models ?? []).join(", ")}{e.n_items ? ` · ${e.n_items} items` : ""}</div></li>
        ))}</ul> : <p className="text-sm text-muted">None yet.</p>}
      </Card>
    </div>
  );
}
