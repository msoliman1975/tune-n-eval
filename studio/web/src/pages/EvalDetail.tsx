import { useMutation, useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { api, fmtMetric, type EvalItem, type EvalProgress, type EvalReport, type LegacyReport } from "../api";
import { BarChart } from "../BarChart";
import { Badge, Button, Card, Code, ErrorNote, Field, inputCls, Pager, ResultTable, Swatch, Tabs, when } from "../ui";
import { LegacyEvalDetail } from "./LegacyEvalDetail";

const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const color = (i: number) => css(`--color-series-${Math.min(i, 7) + 1}`);
const HARDNESS = ["easy", "medium", "hard", "extra"];
const VERDICT: Record<string, string> = { a_better: "A better", b_better: "B better", no_clear_difference: "no clear difference" };

export function EvalDetail({ id }: { id: string }) {
  const rep = useQuery({
    queryKey: ["eval", id], queryFn: () => api.evalReport(id),
    refetchInterval: (q) => (["succeeded", "failed", "cancelled"].includes(q.state.data?.status ?? "") ? false : 4000),
  });
  const r = rep.data;
  if (!r) return <Card><ErrorNote error={rep.error} /></Card>;
  if (r.status !== "succeeded") return <Progress id={id} p={r as EvalProgress} />;
  if ((r as EvalReport).version !== 2) return <LegacyEvalDetail id={id} r={r as LegacyReport} />;
  return <Report r={r as EvalReport} />;
}

function Progress({ id, p }: { id: string; p: EvalProgress }) {
  const cancel = useMutation({ mutationFn: () => api.cancelEval(id) });
  return (
    <Card title={id} actions={<div className="flex items-center gap-3"><Badge value={p.status} />
      {["queued", "running"].includes(p.status) && <Button tone="plain" onClick={() => cancel.mutate()}>Cancel</Button>}</div>}>
      {p.spec && <p className="mb-3 text-sm text-muted">{p.spec.models.map((m) => m.id).join(", ")} on {p.spec.dataset} · {p.estimate?.n_items} items · {p.spec.metrics.join(", ")}</p>}
      <div className="grid gap-2">
        {(p.progress?.stages ?? []).map((s, i) => (
          <div key={i} className="text-sm">
            <div className="flex justify-between gap-2"><span>{s.stage}{s.model ? ` · ${s.model}` : ""}{s.message ? ` · ${s.message}` : ""}</span><span className="num text-muted">{s.done ?? 0} / {s.total ?? "?"}</span></div>
            {s.total ? <div className="h-1.5 rounded bg-bg"><div className="h-1.5 rounded bg-accent" style={{ width: `${Math.min(100, (100 * (s.done ?? 0)) / s.total)}%` }} /></div> : null}
          </div>
        ))}
      </div>
      {p.status === "queued" && <p className="text-sm text-muted">Waiting for the GPU (a training run or another evaluation holds it).</p>}
      <pre className="mt-3 overflow-x-auto font-mono text-xs text-muted">{[...(p.progress?.log ?? []), ...(p.status === "failed" ? p.log_tail : [])].join("\n")}</pre>
    </Card>
  );
}

function Report({ r }: { r: EvalReport }) {
  const models = r.models.map((m) => m.id);
  const info = r.metric_info;
  const numeric = r.metrics.filter((m) => info[m].kind !== "category");
  const primary = numeric[0];
  const slices = useMemo(() => {
    const keys = [...new Set(r.scores.map((s) => `${s.slice.key}:${s.slice.value}`))];
    const order = (k: string) => (k === "all:all" ? -1 : k.startsWith("hardness:") ? HARDNESS.indexOf(k.split(":")[1]) + 10 : 50);
    return keys.sort((a, b) => order(a) - order(b) || a.localeCompare(b));
  }, [r]);
  const [slice, setSlice] = useState("all:all");
  const score = (m: string, metric: string, sl = slice) => r.scores.find((s) => s.model === m && s.metric === metric && `${s.slice.key}:${s.slice.value}` === sl);
  const chartSlices = slices.filter((s) => s.startsWith("hardness:"));
  const byCat = chartSlices.length ? chartSlices : slices.filter((s) => s.startsWith("split:"));
  const name = (id: string) => r.models.find((m) => m.id === id)?.name ?? id;

  return (
    <div className="grid gap-6">
      <div>
        <a href="#/evals" className="text-sm text-muted">← Evaluations</a>
        <h1 className="text-2xl font-bold">{r.id}</h1>
        <p className="text-sm text-muted">{r.task} · {r.dataset} ({r.splits.join(", ")}) · {r.n_items} items{r.limit ? ` (sample, seed ${r.seed})` : ""} · {when(r.created_at)}
          {r.judge?.model ? ` · judge ${r.judge.model}` : ""}</p>
      </div>

      <Card title="Scorecard" actions={<Tabs value={slice} options={slices.map((s) => [s, s === "all:all" ? "all" : s.split(":")[1]] as [string, string])} onChange={setSlice} />}>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left font-mono text-xs uppercase text-muted">
              <tr><th className="py-2">Metric</th>{models.map((m, i) => <th key={m} className="pr-3 text-right"><span className="inline-flex items-center gap-1 normal-case"><Swatch i={i} />{name(m)}</span></th>)}</tr>
            </thead>
            <tbody>
              {numeric.map((metric) => {
                const vals = models.map((m) => score(m, metric)?.value);
                const present = vals.filter((v): v is number => v !== undefined);
                const best = present.length > 1 && info[metric].kind !== "value" ? (info[metric].higher_is_better ? Math.max(...present) : Math.min(...present)) : undefined;
                return (
                  <tr key={metric} className="border-t border-line">
                    <td className="py-2"><span className="font-medium">{info[metric].name}</span><div className="text-xs text-muted">{info[metric].approach.replace("_", " ")}</div></td>
                    {models.map((m) => {
                      const s = score(m, metric);
                      return (
                        <td key={m} className="pr-3 text-right num">
                          <span className={s && s.value === best ? "font-bold" : ""}>{fmtMetric(info[metric].kind, s?.value)}</span>
                          {s && <div className="text-xs text-muted">[{fmtMetric(info[metric].kind, s.ci95[0])}, {fmtMetric(info[metric].kind, s.ci95[1])}]{s.p90 !== undefined ? ` p90 ${s.p90.toFixed(1)} s` : ""} · n {s.n}</div>}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        {slice === "all:all" && r.baselines && Object.keys(r.baselines).length > 0 && (
          <p className="mt-2 text-xs text-muted">Guessing baselines, the best a constant answer scores on these items: {Object.entries(r.baselines).map(([m, b]) => `${info[m]?.name ?? m} ${fmtMetric(info[m]?.kind, b.value)} (${b.answer})`).join(" · ")}. A model near these numbers is not judging.</p>
        )}
        <p className="mt-2 text-xs text-muted">95% bootstrap intervals. Bold = best on this slice. Hardness is Spider's classification of the gold query.</p>
      </Card>

      {byCat.length > 0 && primary && (
        <Card title={`${info[primary].name} by ${chartSlices.length ? "difficulty" : "split"}`}>
          <BarChart categories={byCat.map((s) => s.split(":")[1])} max={["rate", "score"].includes(info[primary].kind) ? 1 : undefined}
                    format={(v) => fmtMetric(info[primary].kind, v)}
                    series={models.map((m, i) => ({ name: name(m), color: color(i), values: byCat.map((s) => score(m, primary, s)?.value ?? null) }))} />
          <p className="text-xs text-muted">Items per group: {byCat.map((s) => `${s.split(":")[1]} ${score(models[0], primary, s)?.n ?? 0}`).join(" · ")}. Exact values are in the scorecard.</p>
        </Card>
      )}

      {models.length > 1 && <Comparisons r={r} slice={slice} name={name} />}
      {r.breakdowns.sql_failure_judge && <Failures r={r} name={name} />}
      <Usage r={r} name={name} />
      <Explorer r={r} name={name} />
    </div>
  );
}

function Comparisons({ r, slice, name }: { r: EvalReport; slice: string; name: (id: string) => string }) {
  const numeric = r.metrics.filter((m) => r.metric_info[m].kind !== "category");
  const [metric, setMetric] = useState(numeric[0]);
  const kind = r.metric_info[metric].kind;
  const rows = r.comparisons.filter((c) => c.metric === metric && `${c.slice.key}:${c.slice.value}` === slice);
  const d = (x: number) => (kind === "rate" || kind === "score" ? `${(x * 100).toFixed(1)}` : x.toFixed(2));
  return (
    <Card title="Pairwise comparisons" actions={<select className={inputCls} value={metric} onChange={(e) => setMetric(e.target.value)}>{numeric.map((m) => <option key={m} value={m}>{r.metric_info[m].name}</option>)}</select>}>
      <p className="mb-3 text-sm text-muted">Paired on the same items: Δ = B − A with a 95% bootstrap interval{kind === "rate" || kind === "score" ? " in percentage points" : ""}. An interval that includes 0 is “no clear difference”.</p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted"><tr><th className="py-2">A</th><th>B</th><th className="text-right">Δ [95% CI]</th><th className="text-right">n</th><th className="pl-4">Verdict</th></tr></thead>
          <tbody>
            {rows.map((c, i) => (
              <tr key={i} className="border-t border-line">
                <td className="py-2">{name(c.a)}</td><td>{name(c.b)}</td>
                <td className="num text-right">{c.delta >= 0 ? "+" : ""}{d(c.delta)} [{d(c.ci95[0])}, {d(c.ci95[1])}]</td>
                <td className="num text-right text-muted">{c.n}</td>
                <td className="pl-4"><Badge value={c.verdict === "no_clear_difference" ? "no_clear_difference" : "ok"} label={c.verdict === "no_clear_difference" ? VERDICT[c.verdict] : `${name(c.verdict === "a_better" ? c.a : c.b)} better`} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Failures({ r, name }: { r: EvalReport; name: (id: string) => string }) {
  const b = r.breakdowns.sql_failure_judge;
  const models = Object.keys(b);
  const cats = [...new Set(models.flatMap((m) => Object.keys(b[m])))].sort((x, y) => models.reduce((s, m) => s + (b[m][y] ?? 0) - (b[m][x] ?? 0), 0));
  return (
    <Card title="Why answers fail" actions={<span className="text-xs text-muted">judge {r.judge.model} · one main category per failed answer</span>}>
      <BarChart horizontal integer categories={cats.map((c) => c.replace(/_/g, " "))} height={Math.max(180, cats.length * 34 + 40)} format={(v) => String(Math.round(v))}
                series={models.map((m) => ({ name: name(m), color: color(r.models.findIndex((x) => x.id === m)), values: cats.map((c) => b[m][c] ?? 0) }))} />
      <details className="mt-2"><summary className="cursor-pointer text-xs text-muted">Table</summary>
        <table className="mt-2 text-sm"><thead className="font-mono text-xs uppercase text-muted"><tr><th className="pr-4 text-left">category</th>{models.map((m) => <th key={m} className="pr-4 text-right">{name(m)}</th>)}</tr></thead>
          <tbody>{cats.map((c) => <tr key={c} className="border-t border-line"><td className="pr-4">{c.replace(/_/g, " ")}</td>{models.map((m) => <td key={m} className="num pr-4 text-right">{b[m][c] ?? 0}</td>)}</tr>)}</tbody></table>
      </details>
    </Card>
  );
}

function Usage({ r, name }: { r: EvalReport; name: (id: string) => string }) {
  return (
    <Card title="Tokens and cost">
      <div className="overflow-x-auto">
        <table className="w-full min-w-[520px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted"><tr><th className="py-2">Model</th><th className="text-right">Input tokens</th><th className="text-right">Output tokens</th><th className="text-right">Errors</th><th className="text-right">Cost</th></tr></thead>
          <tbody>
            {r.models.map((m) => {
              const u = r.usage[m.id];
              return (
                <tr key={m.id} className="border-t border-line">
                  <td className="py-2">{name(m.id)}</td><td className="num text-right">{u.input_tokens.toLocaleString()}</td><td className="num text-right">{u.output_tokens.toLocaleString()}</td>
                  <td className={`num text-right ${u.errors ? "text-warn" : ""}`}>{u.errors}</td><td className="num text-right">{u.usd === null ? (m.paid ? "?" : "free") : `$${u.usd.toFixed(2)}`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

function Explorer({ r, name }: { r: EvalReport; name: (id: string) => string }) {
  const numeric = r.metrics.filter((m) => ["rate", "score", "scale"].includes(r.metric_info[m].kind));
  const [metric, setMetric] = useState(numeric[0] ?? r.metrics[0]);
  const [model, setModel] = useState(r.models[0].id);
  const [outcome, setOutcome] = useState("all");
  const [slice, setSlice] = useState("");
  const [q, setQ] = useState("");
  const [offset, setOffset] = useState(0);
  const items = useQuery({ queryKey: ["eval-items", r.id, metric, model, outcome, slice, q, offset],
    queryFn: () => api.evalItems(r.id, { metric, model, outcome, slice, q, offset, limit: 10 }) });
  const sliceOpts = [...new Set(r.scores.filter((s) => s.slice.key !== "all").map((s) => `${s.slice.key}:${s.slice.value}`))];
  const reset = <T,>(f: (v: T) => void) => (v: T) => { f(v); setOffset(0); };
  return (
    <Card title="Answers">
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <Field label="Show"><select className={inputCls} value={outcome} onChange={(e) => reset(setOutcome)(e.target.value)}>
          <option value="all">all items</option><option value="fail">failed by…</option><option value="pass">passed by…</option><option value="disagree">models disagree</option></select></Field>
        {(outcome === "fail" || outcome === "pass") && <Field label="Model"><select className={inputCls} value={model} onChange={(e) => reset(setModel)(e.target.value)}>{r.models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}</select></Field>}
        <Field label="On"><select className={inputCls} value={metric} onChange={(e) => reset(setMetric)(e.target.value)}>{numeric.map((m) => <option key={m} value={m}>{r.metric_info[m].name}</option>)}</select></Field>
        <Field label="Slice"><select className={inputCls} value={slice} onChange={(e) => reset(setSlice)(e.target.value)}><option value="">all</option>{sliceOpts.map((s) => <option key={s} value={s}>{s.replace(":", ": ")}</option>)}</select></Field>
        <Field label="Search"><input className={`${inputCls} w-56`} value={q} onChange={(e) => reset(setQ)(e.target.value)} /></Field>
        <Pager total={items.data?.total ?? 0} offset={offset} limit={10} onChange={setOffset} />
      </div>
      <ErrorNote error={items.error} />
      <div className="grid gap-3">{(items.data?.items ?? []).map((it) => <ItemCard key={it.id} r={r} it={it} metric={metric} name={name} />)}</div>
    </Card>
  );
}

function PromptView({ it, r, name }: { it: EvalItem; r: EvalReport; name: (id: string) => string }) {
  const [tab, setTab] = useState("messages");
  const [copied, setCopied] = useState(false);
  const sent = tab === "messages" ? undefined : it.as_sent?.[tab];
  const text = tab === "messages"
    ? (it.messages ?? []).map((m) => `### ${m.role}\n${m.content}`).join("\n\n")
    : sent?.format === "json" ? JSON.stringify(sent.body, null, 2) : sent?.text ?? sent?.label ?? "";
  const tokens = tab === "messages" ? undefined : it.results[tab]?.input_tokens;
  const copy = () => navigator.clipboard?.writeText(text).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500); });
  return (
    <div className="min-w-0 rounded-md border border-line p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <span className="font-mono text-xs uppercase text-muted">Prompt</span>
        <Tabs value={tab} onChange={setTab}
              options={[["messages", "Messages"], ...r.models.filter((m) => it.as_sent?.[m.id]).map((m): [string, string] => [m.id, `As sent: ${name(m.id)}`])]} />
      </div>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2 text-xs text-muted">
        <span>
          {tab === "messages"
            ? <>The chat messages every model got{it.prompt_source === "re-rendered" && <span className="text-warn"> · re-rendered from the task and dataset (this evaluation predates stored prompts)</span>}</>
            : sent?.label}
          {" · "}<span className="num">{text.length.toLocaleString()}</span> chars
          {typeof tokens === "number" && <> · <span className="num">{tokens.toLocaleString()}</span> input tokens measured</>}
        </span>
        <Button onClick={copy}>{copied ? "Copied" : "Copy"}</Button>
      </div>
      {sent?.format === "error" ? <p className="text-xs text-warn">{sent.label}</p>
        : tab === "messages" ? (it.messages ?? []).map((m, i) => (
          <div key={i} className="mt-2 first:mt-0"><div className="font-mono text-xs uppercase text-muted">{m.role}</div><Code className="max-h-96 overflow-y-auto">{m.content}</Code></div>
        ))
        : <Code className="max-h-96 overflow-y-auto">{text}</Code>}
    </div>
  );
}

function ItemCard({ r, it, metric, name }: { r: EvalReport; it: EvalItem; metric: string; name: (id: string) => string }) {
  const [open, setOpen] = useState(false);
  const full = useQuery({ queryKey: ["eval-item", r.id, it.id], queryFn: () => api.evalItem(r.id, it.id), enabled: open });
  const sqlTask = r.metrics.some((m) => m.startsWith("sql_"));
  return (
    <div className="rounded-md border border-line p-3">
      <button className="flex w-full flex-wrap items-start justify-between gap-2 text-left" onClick={() => setOpen(!open)} aria-expanded={open}>
        <span className="min-w-0"><span dir="auto" className="font-medium">{it.inputs.question ?? Object.values(it.inputs)[0]}</span>
          <span className="block font-mono text-xs text-muted">{it.id} · {Object.entries(it.slices).map(([k, v]) => `${k} ${v}`).join(" · ")}{it.inputs.db_id ? ` · db ${it.inputs.db_id}` : ""}</span></span>
        <span className="flex flex-wrap gap-1">
          {r.models.map((m, i) => {
            const v = it.results[m.id]?.scores[metric];
            const ok = typeof v === "number" && ["rate", "score", "scale"].includes(r.metric_info[metric].kind) ? (r.metric_info[metric].kind === "scale" ? v >= 4 : v >= (r.metric_info[metric].kind === "score" ? 0.5 : 1)) : null;
            return <span key={m.id} className="inline-flex items-center gap-1"><Swatch i={i} /><Badge value={ok === null ? "unknown" : ok ? "pass" : "fail"} label={typeof v === "number" && r.metric_info[metric].kind !== "rate" ? fmtMetric(r.metric_info[metric].kind, v) : ok === null ? "—" : ok ? "pass" : "fail"} /></span>;
          })}
        </span>
      </button>
      {open && (
        <div className="mt-3 grid gap-3">
          <div className="grid gap-3 md:grid-cols-2">
            <div className="min-w-0"><div className="font-mono text-xs uppercase text-muted">reference</div><Code>{it.reference}</Code>{sqlTask && <ResultTable r={it.gold_result} max={5} />}</div>
            {r.models.map((m, i) => {
              const x = it.results[m.id];
              if (!x) return null;
              const judge = x.details.sql_failure_judge ?? x.details.llm_judge;
              return (
                <div key={m.id} className="min-w-0">
                  <div className="flex items-center gap-1 font-mono text-xs uppercase text-muted"><Swatch i={i} />{name(m.id)} · {x.latency_s.toFixed(2)} s</div>
                  <Code>{x.details.sql ?? x.output}</Code>
                  {x.error && <p className="text-xs text-warn">{x.error}</p>}
                  {sqlTask && <ResultTable r={x.details.result} max={5} />}
                  <p className="mt-1 text-xs text-muted">{Object.entries(x.scores).map(([k, v]) => `${r.metric_info[k]?.name ?? k}: ${typeof v === "number" ? fmtMetric(r.metric_info[k]?.kind, v) : String(v).replace(/_/g, " ")}`).join(" · ")}</p>
                  {judge && "rationale" in judge && judge.rationale && <p className="mt-1 text-xs italic text-muted">{judge.rationale}</p>}
                  {x.details.sql && x.output.trim() !== x.details.sql && <details><summary className="cursor-pointer text-xs text-muted">raw output</summary><Code>{x.output}</Code></details>}
                </div>
              );
            })}
          </div>
          {full.data ? <PromptView it={full.data} r={r} name={name} /> : <p className="text-xs text-muted">Loading the prompt…</p>}
        </div>
      )}
    </div>
  );
}
