import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, ErrorNote, Field, inputCls } from "../ui";
import { where } from "./Models";

const APPROACH_ORDER = ["execution", "reference", "structured", "product", "llm_judge", "efficiency"];
const APPROACH_LABEL: Record<string, string> = { execution: "Execution-based", reference: "Reference match", llm_judge: "LLM judge", efficiency: "Efficiency" };

export function EvalNew({ query }: { query: URLSearchParams }) {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
  const [dataset, setDataset] = useState(query.get("dataset") ?? "");
  const [picked, setPicked] = useState<string[]>((query.get("models") ?? "").split(",").filter(Boolean));
  const [splits, setSplits] = useState<string[]>([]);
  const [limit, setLimit] = useState(0);
  const [seed, setSeed] = useState(7);
  const [metrics, setMetrics] = useState<string[]>([]);
  const [judge, setJudge] = useState("qwen3:14b");
  const [confirmPaid, setConfirmPaid] = useState(false);

  const generic = datasets.data ?? [];
  const ds = datasets.data?.find((d) => d.path === dataset);
  const task = tasks.data?.find((t) => t.ref === ds?.task);
  useEffect(() => { if (!dataset && generic[0]) setDataset(generic.find((d) => d.task === "nl2sql-sqlite@v1")?.path ?? generic[0].path); }, [datasets.data]);
  useEffect(() => {
    if (!ds) return;
    const s = Object.keys(ds.splits);
    setSplits(s.includes("test") ? ["test"] : s.slice(-1));
  }, [ds?.path]);
  const catalog = useQuery({ queryKey: ["metrics", task?.ref], queryFn: () => api.metrics(task!.ref), enabled: !!task });
  useEffect(() => { if (catalog.data) setMetrics(catalog.data.metrics.filter((m) => m.default).map((m) => m.id)); }, [catalog.data]);

  const body = { models: picked, dataset, splits, limit, seed, metrics, judge_model: judge, confirm_paid: confirmPaid };
  const ready = picked.length > 0 && !!dataset && splits.length > 0 && metrics.length > 0;
  const estimate = useQuery({ queryKey: ["eval-estimate", body.models, dataset, splits, limit], queryFn: () => api.evalEstimate(body), enabled: ready });
  const go = useMutation({ mutationFn: () => api.evaluate(body), onSuccess: (r) => { location.hash = `#/evals/${r.id}`; } });
  const paid = picked.filter((id) => models.data?.find((m) => m.id === id)?.paid);
  const cost = Object.values(estimate.data?.models ?? {}).reduce((s, e) => s + (e.est_usd ?? 0), 0);
  const ollama = (models.data ?? []).filter((m) => m.kind === "ollama");
  const grouped = useMemo(() => APPROACH_ORDER.map((a) => [a, (catalog.data?.metrics ?? []).filter((m) => m.approach === a)] as const).filter(([, ms]) => ms.length), [catalog.data]);
  const toggle = (list: string[], set: (v: string[]) => void, v: string) => set(list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  return (
    <div className="grid gap-6">
      <div><a href="#/evals" className="text-sm text-muted">← Evaluations</a><h1 className="text-2xl font-bold">New evaluation</h1></div>

      <Card title="1 · Dataset">
        <div className="flex flex-wrap items-end gap-4">
          <Field label="Dataset" hint={task?.name}>
            <select className={inputCls} value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {generic.map((d) => <option key={d.path} value={d.path}>{d.id}@{d.version}</option>)}
            </select>
          </Field>
          <Field label="Splits">
            <div className="flex gap-3">{Object.keys(ds?.splits ?? {}).map((s) => (
              <label key={s} className="flex items-center gap-1 text-sm"><input type="checkbox" checked={splits.includes(s)} onChange={() => toggle(splits, setSplits, s)} />{s} <span className="num text-muted">{ds?.counts[s]}</span></label>
            ))}</div>
          </Field>
          <Field label="Sample (0 = all)" hint="the same items for every model"><input type="number" min={0} className={`${inputCls} w-28`} value={limit} onChange={(e) => setLimit(+e.target.value)} /></Field>
          <Field label="Seed"><input type="number" className={`${inputCls} w-20`} value={seed} onChange={(e) => setSeed(+e.target.value)} /></Field>
        </div>
      </Card>

      <Card title={`2 · Models (${picked.length})`}>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {(models.data ?? []).map((m) => (
            <label key={m.id} className={`flex cursor-pointer gap-2 rounded-md border p-2 text-sm ${picked.includes(m.id) ? "border-accent bg-accent-soft" : "border-line"}`}>
              <input type="checkbox" checked={picked.includes(m.id)} onChange={() => toggle(picked, setPicked, m.id)} />
              <span className="min-w-0"><span className="font-medium">{m.name}</span> {m.paid && <Badge value="paid" />}
                <span className="block truncate font-mono text-xs text-muted">{where(m)}{m.kind === "hf_adapter" && m.tags.length ? ` · ${m.tags.filter((t) => t !== "fine-tuned").join(", ")}` : ""}</span></span>
            </label>
          ))}
        </div>
        <p className="mt-2 text-xs text-muted">Models are compared pairwise on the same items. Local GPU models run first, then Ollama, then APIs; answers are cached, so re-running or adding a model only generates what is missing.</p>
      </Card>

      <Card title="3 · Metrics">
        {!task ? <p className="text-sm text-muted">Pick a dataset.</p> : (
          <div className="grid gap-4 md:grid-cols-2">
            {grouped.map(([a, ms]) => (
              <div key={a}>
                <h3 className="mb-1 font-bold">{APPROACH_LABEL[a] ?? a}</h3>
                <p className="mb-2 text-xs text-muted">{catalog.data?.approaches[a]?.split(": ").slice(1).join(": ")}</p>
                {ms.map((m) => (
                  <label key={m.id} className="mb-2 flex gap-2 text-sm">
                    <input type="checkbox" className="mt-1" checked={metrics.includes(m.id)} onChange={() => toggle(metrics, setMetrics, m.id)} />
                    <span><span className="font-medium">{m.name}</span><span className="block text-xs text-muted">{m.description}</span></span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        )}
        {metrics.some((m) => catalog.data?.metrics.find((x) => x.id === m)?.judge) && (
          <div className="mt-2"><Field label="Judge model (local, free)" hint="pick a family different from the models under test">
            <select className={`${inputCls} w-64`} value={judge} onChange={(e) => setJudge(e.target.value)}>
              {ollama.map((m) => <option key={m.id} value={m.ollama_tag!}>{m.name}</option>)}
            </select></Field></div>
        )}
      </Card>

      <Card title="4 · Run">
        <p className="text-sm text-muted">
          {estimate.data ? <><b className="num text-fg">{estimate.data.n_items}</b> items × <b className="num text-fg">{picked.length}</b> models.</> : "…"}
          {paid.length > 0 && estimate.data && <> Paid: {paid.join(", ")}, about <b className="text-fg">${cost.toFixed(2)}</b> (estimate; Claude thinking can add more).</>}
        </p>
        {paid.length > 0 && <label className="mt-2 flex items-center gap-2 text-sm"><input type="checkbox" checked={confirmPaid} onChange={(e) => setConfirmPaid(e.target.checked)} />I accept the API cost</label>}
        <div className="mt-3"><Button disabled={!ready || go.isPending || (paid.length > 0 && !confirmPaid)} onClick={() => go.mutate()}>Start evaluation</Button></div>
        <ErrorNote error={go.error ?? estimate.error} />
      </Card>
    </div>
  );
}
