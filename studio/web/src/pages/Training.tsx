import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, ErrorNote, Field, inputCls, when } from "../ui";

export function Training() {
  const qc = useQueryClient();
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs, refetchInterval: 5000 });

  const [preset, setPreset] = useState("qwen-8b-qlora");
  const [dataset, setDataset] = useState("");
  const [epochs, setEpochs] = useState(2);
  const [maxSteps, setMaxSteps] = useState(-1);
  useEffect(() => {
    const real = datasets.data?.find((d) => d.id !== "fixture") ?? datasets.data?.[0];
    if (!dataset && real) setDataset(real.path);
  }, [datasets.data, dataset]);

  const body = { preset, dataset, epochs, max_steps: maxSteps };
  const estimate = useQuery({
    queryKey: ["estimate", body], queryFn: () => api.estimate(body), enabled: !!dataset,
  });
  const launch = useMutation({
    mutationFn: () => api.launch(body),
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["runs"] }); location.hash = `#/runs/${r.id}`; },
  });

  return (
    <div className="grid gap-6">
      <Card title="Launch a fine-tuning run">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Field label="Preset">
            <select className={inputCls} value={preset} onChange={(e) => setPreset(e.target.value)}>
              {Object.entries(presets.data ?? {}).map(([k, p]) => (
                <option key={k} value={k}>{k} · {p.measured_vram_gb} GB</option>
              ))}
            </select>
          </Field>
          <Field label="Dataset">
            <select className={inputCls} value={dataset} onChange={(e) => setDataset(e.target.value)}>
              {(datasets.data ?? []).map((d) => (
                <option key={d.path} value={d.path}>{d.id}@{d.version} · {d.counts.train ?? 0} train</option>
              ))}
            </select>
          </Field>
          <Field label="Epochs">
            <input type="number" min={0.1} step={0.5} className={inputCls} value={epochs} onChange={(e) => setEpochs(+e.target.value)} />
          </Field>
          <Field label="Max steps (−1 = full)">
            <input type="number" className={inputCls} value={maxSteps} onChange={(e) => setMaxSteps(+e.target.value)} />
          </Field>
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted">
            {estimate.data ? (
              <>
                VRAM <b className="text-fg num">{estimate.data.vram_gb}</b> of {estimate.data.vram_total_gb} GB measured in M0 ·
                {" "}<b className="text-fg num">{estimate.data.train_rows}</b> training rows · about
                {" "}<b className="text-fg num">{estimate.data.est_minutes}</b> min
                {!estimate.data.fits && <span className="text-warn"> · will not fit</span>}
              </>
            ) : datasets.data?.length === 0 ? "No frozen dataset yet: run the Data pipeline first." : "…"}
          </p>
          <Button disabled={!dataset || launch.isPending || estimate.data?.fits === false} onClick={() => launch.mutate()}>
            {launch.isPending ? "Queuing…" : "Launch run"}
          </Button>
        </div>
        <ErrorNote error={launch.error ?? estimate.error} />
      </Card>

      <Card title="Runs">
        <ErrorNote error={runs.error} />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[640px] text-sm">
            <thead className="text-left font-mono text-xs uppercase text-muted">
              <tr><th className="py-2">Run</th><th>Status</th><th>Progress</th><th>Train loss</th><th>Eval loss</th><th>Started</th></tr>
            </thead>
            <tbody>
              {(runs.data ?? []).map((r) => (
                <tr key={r.id} className="border-t border-line">
                  <td className="py-2"><a className="text-accent" href={`#/runs/${r.id}`}>{r.id}</a></td>
                  <td><Badge value={r.status} /></td>
                  <td className="num">{r.total ? `${r.step} / ${r.total}` : "—"}</td>
                  <td className="num">{r.result?.train_loss?.toFixed(3) ?? "—"}</td>
                  <td className="num">{r.result?.eval_loss?.toFixed(3) ?? "—"}</td>
                  <td className="text-muted">{when(r.created_at)}</td>
                </tr>
              ))}
              {runs.data?.length === 0 && <tr><td colSpan={6} className="py-6 text-center text-muted">No runs yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
