import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, ErrorNote, Field, inputCls, when } from "../ui";
import { TagEditor } from "./Models";

export function Training() {
  const qc = useQueryClient();
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const presets = useQuery({ queryKey: ["presets"], queryFn: api.presets });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs, refetchInterval: 5000 });
  const trainable = (models.data ?? []).filter((m) => m.trainable);

  const [model, setModel] = useState(new URLSearchParams(location.hash.split("?")[1] ?? "").get("model") ?? "gemma-4-e4b");
  const [dataset, setDataset] = useState(new URLSearchParams(location.hash.split("?")[1] ?? "").get("dataset") ?? "");
  const [epochs, setEpochs] = useState(1);
  const [maxSteps, setMaxSteps] = useState(-1);
  const [percent, setPercent] = useState(100);
  useEffect(() => {
    const real = datasets.data?.find((d) => d.id !== "fixture" && d.counts.train) ?? datasets.data?.[0];
    if (!dataset && real) setDataset(real.path);
  }, [datasets.data, dataset]);

  const chosen = trainable.find((m) => m.id === model);
  const preset = chosen?.train_preset ? presets.data?.[chosen.train_preset] : undefined;
  const ds = datasets.data?.find((d) => d.path === dataset);
  const share = Math.min(100, Math.max(1, Math.round(percent || 100)));
  const body = { model, dataset, epochs, max_steps: maxSteps, train_fraction: share / 100 };
  const estimate = useQuery({ queryKey: ["estimate", body], queryFn: () => api.estimate(body), enabled: !!dataset && !!chosen });
  const [name, setName] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const knownTags = [...new Set((models.data ?? []).flatMap((m) => m.tags))].sort();
  const defaultName = chosen && ds ? `${chosen.name.split(" (")[0]} · ${ds.id}@${ds.version}${share < 100 ? ` · ${share}%` : ""} · <launch time>` : "";
  const launch = useMutation({
    mutationFn: () => api.launch({ ...body, name: name.trim() || null, tags }),
    onSuccess: (r) => { qc.invalidateQueries({ queryKey: ["runs"] }); location.hash = `#/runs/${r.id}`; },
  });
  const e = estimate.data;

  return (
    <div className="grid gap-6">
      <Card title="Launch a fine-tuning run">
        <div className="grid items-start gap-4 sm:grid-cols-2 lg:grid-cols-5">
          <Field label="Base model" hint={preset ? `${chosen?.train_preset} · ${preset.load_in_4bit ? "QLoRA 4-bit" : "LoRA"} · seq ${preset.seq}` : undefined}>
            <select className={inputCls} value={model} onChange={(ev) => setModel(ev.target.value)}>
              {trainable.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
            </select>
          </Field>
          <Field label="Dataset" hint={ds ? tasks.data?.find((t) => t.ref === ds.task)?.name : undefined}>
            <select className={inputCls} value={dataset} onChange={(ev) => setDataset(ev.target.value)}>
              {(datasets.data ?? []).filter((d) => d.counts.train).map((d) => (
                <option key={d.path} value={d.path}>{d.id}@{d.version} · {d.counts.train} train</option>
              ))}
            </select>
          </Field>
          <Field label="Training data (%)" hint={e && share < 100 ? `≈${e.train_rows} of ${e.total_train_rows} rows, stratified` : "share of the train split"}>
            <input type="number" min={1} max={100} step={5} className={inputCls} value={percent} onChange={(ev) => setPercent(+ev.target.value)} />
          </Field>
          <Field label="Epochs"><input type="number" min={0.1} step={0.5} className={inputCls} value={epochs} onChange={(ev) => setEpochs(+ev.target.value)} /></Field>
          <Field label="Max steps (−1 = full)"><input type="number" className={inputCls} value={maxSteps} onChange={(ev) => setMaxSteps(+ev.target.value)} /></Field>
        </div>
        <div className="mt-4 grid items-start gap-4 sm:grid-cols-2">
          <Field label="Name of the fine-tune (optional)" hint="how it appears in Models, evaluations and the playground; editable later">
            <input className={inputCls} value={name} placeholder={defaultName} onChange={(ev) => setName(ev.target.value)} />
          </Field>
          <Field label="Tags (optional)" hint={`"fine-tuned" and the dataset id are added automatically`}>
            <TagEditor tags={tags} onChange={setTags} suggestions={knownTags} />
          </Field>
        </div>
        <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm text-muted">
            {e ? (
              <>
                {e.vram_gb !== null ? <>VRAM <b className="num text-fg">{e.vram_gb}</b> of {e.vram_total_gb} GB measured · </> : <>VRAM not benchmarked yet for this preset · </>}
                <b className="num text-fg">{e.train_rows}</b> training rows of ~<b className="num text-fg">{e.avg_tokens}</b> tokens ·
                {e.est_minutes !== null ? <> about <b className="num text-fg">{e.est_minutes}</b> min</> : " duration unknown until benchmarked"}
                {e.fits === false && <span className="text-warn"> · will not fit</span>}
              </>
            ) : datasets.data?.length === 0 ? "No frozen dataset yet: create one under Datasets." : "…"}
          </p>
          <Button disabled={!dataset || !chosen || launch.isPending || e?.fits === false} onClick={() => launch.mutate()}>
            {launch.isPending ? "Queuing…" : "Launch run"}
          </Button>
        </div>
        <ErrorNote error={launch.error ?? estimate.error} />
      </Card>

      <Card title="Runs">
        <ErrorNote error={runs.error} />
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="text-left font-mono text-xs uppercase text-muted">
              <tr><th className="py-2">Run</th><th>Status</th><th>Progress</th><th>Train loss</th><th>Eval loss</th><th>Ollama</th><th>Started</th></tr>
            </thead>
            <tbody>
              {(runs.data ?? []).map((r) => (
                <tr key={r.id} className="border-t border-line">
                  <td className="py-2">
                    {(() => { const m = models.data?.find((x) => x.id === r.id); return m ? <><a className="text-accent" href={`#/runs/${r.id}`}>{m.name}</a><div className="font-mono text-xs text-muted">{r.id}</div></>
                      : <a className="text-accent" href={`#/runs/${r.id}`}>{r.id}</a>; })()}
                    <div className="text-xs text-muted">{r.message}</div>
                  </td>
                  <td><Badge value={r.status} /></td>
                  <td className="num">{r.total ? `${r.step} / ${r.total}` : "—"}</td>
                  <td className="num">{r.result?.train_loss?.toFixed(3) ?? "—"}</td>
                  <td className="num">{r.result?.eval_loss?.toFixed(3) ?? "—"}</td>
                  <td className="font-mono text-xs">{r.export?.ollama_tag ?? (r.export ? <Badge value={r.export.status} /> : "—")}</td>
                  <td className="text-muted">{when(r.created_at)}</td>
                </tr>
              ))}
              {runs.data?.length === 0 && <tr><td colSpan={7} className="py-6 text-center text-muted">No runs yet.</td></tr>}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
