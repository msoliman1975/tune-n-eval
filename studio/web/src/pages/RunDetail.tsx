import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { api, type JobEvent } from "../api";
import { LineChart } from "../LineChart";
import { Badge, Button, Card, ErrorNote, Field, inputCls } from "../ui";

const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

/** Live view of one run: JobEvents stream in over SSE while it trains. */
export function RunDetail({ id }: { id: string }) {
  const run = useQuery({ queryKey: ["run", id], queryFn: () => api.run(id), refetchInterval: 10000 });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [status, setStatus] = useState<string>("");

  useEffect(() => {
    setEvents([]);
    const es = new EventSource(`/api/train/runs/${id}/events`);
    es.onmessage = (m) => {
      const ev: JobEvent = JSON.parse(m.data);
      setStatus(ev.type === "metric" || ev.type === "log" ? "running" : ev.type);
      if (ev.type === "metric") setEvents((prev) => [...prev, ev]);
      if (["succeeded", "failed", "cancelled"].includes(ev.type)) { es.close(); run.refetch(); }
    };
    es.onerror = () => es.close();
    return () => es.close();
  }, [id]);

  const series = useMemo(() => {
    const pts = (k: string) => events.filter((e) => e.metrics?.[k] !== undefined).map((e) => [e.step ?? 0, e.metrics![k]] as [number, number]);
    return {
      loss: [
        { name: "train loss", data: pts("loss"), color: css("--color-accent") },
        { name: "eval loss (reworded questions)", data: pts("eval_loss"), color: css("--color-mango"), dashed: true },
      ],
      gpu: [{ name: "VRAM GB", data: pts("vram_gb"), color: css("--color-accent") }],
      temp: [{ name: "GPU °C", data: pts("gpu_temp_c"), color: css("--color-warn") }],
    };
  }, [events]);

  const last = events.at(-1);
  const r = run.data;
  const shownStatus = r && ["succeeded", "failed", "cancelled"].includes(r.status) ? r.status : status || r?.status;
  const cancel = useMutation({ mutationFn: () => api.cancel(id), onSuccess: () => run.refetch() });

  return (
    <div className="grid gap-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <a href="#/training" className="text-sm text-muted">← Runs</a>
          <h1 className="text-2xl font-bold">{id}</h1>
          <p className="text-sm text-muted">{r?.message}</p>
        </div>
        <div className="flex items-center gap-3">
          <Badge value={shownStatus} />
          {last?.total ? <span className="num text-sm text-muted">step {last.step} / {last.total}</span> : null}
          {(shownStatus === "running" || shownStatus === "queued" || shownStatus === "started") &&
            <Button tone="plain" onClick={() => cancel.mutate()}>Cancel</Button>}
        </div>
      </div>
      <ErrorNote error={run.error} />

      <Card title="Loss">
        {events.length ? <LineChart series={series.loss} yName="loss" /> : <p className="text-sm text-muted">Waiting for the first logged step…</p>}
      </Card>
      <div className="grid gap-6 md:grid-cols-2">
        <Card title="GPU memory"><LineChart series={series.gpu} yName="GB" height={180} /></Card>
        <Card title="GPU temperature"><LineChart series={series.temp} yName="°C" height={180} /></Card>
      </div>

      {r?.result && (
        <Card title="Result">
          <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-5">
            {Object.entries(r.result).map(([k, v]) => (
              <div key={k}><dt className="font-mono text-xs uppercase text-muted">{k.replace(/_/g, " ")}</dt>
                <dd className="num text-lg">{typeof v === "number" ? +v.toFixed(3) : v}</dd></div>
            ))}
          </dl>
        </Card>
      )}
      {r?.model_ref && <EvaluateCard modelRef={`artifacts/train/${id}/model_ref.json`} datasets={datasets.data ?? []}
                                       defaultDataset={r.model_ref.lineage?.dataset} />}
    </div>
  );
}

function EvaluateCard({ modelRef, datasets, defaultDataset }:
  { modelRef: string; datasets: { id: string; version: string; path: string }[]; defaultDataset?: string }) {
  const match = datasets.find((d) => `${d.id}@${d.version}` === defaultDataset);
  const [dataset, setDataset] = useState(match?.path ?? "");
  const [judge, setJudge] = useState(false);
  useEffect(() => { if (!dataset && match) setDataset(match.path); }, [match, dataset]);
  const go = useMutation({
    mutationFn: () => api.evaluate({ model_ref: modelRef, dataset, judge }),
    onSuccess: (r) => { location.hash = `#/evals/${r.id}`; },
  });
  return (
    <Card title="Evaluate against the base model">
      <div className="flex flex-wrap items-end gap-4">
        <Field label="Dataset">
          <select className={inputCls} value={dataset} onChange={(e) => setDataset(e.target.value)}>
            {datasets.map((d) => <option key={d.path} value={d.path}>{d.id}@{d.version}</option>)}
          </select>
        </Field>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={judge} onChange={(e) => setJudge(e.target.checked)} />
          Also judge open answers with local Gemma 3 12B (free, adds time; MCQ and chrF++ always run)
        </label>
        <Button disabled={!dataset || go.isPending} onClick={() => go.mutate()}>Run evaluation</Button>
      </div>
      <ErrorNote error={go.error} />
    </Card>
  );
}
