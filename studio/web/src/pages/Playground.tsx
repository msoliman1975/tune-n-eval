import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, Code, ErrorNote, Field, inputCls, ResultTable, Swatch } from "../ui";

/** Ask one question of several models with the task prompt used in training and evaluation
 * (schema injected for SQL tasks), and run each answer against the database. */
export function Playground() {
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
  const withDbs = (datasets.data ?? []).filter((d) => d.attachments);
  const [dataset, setDataset] = useState("");
  useEffect(() => { if (!dataset && withDbs[0]) setDataset(withDbs[0].path); }, [datasets.data]);
  const ds = withDbs.find((d) => d.path === dataset);
  const task = tasks.data?.find((t) => t.ref === ds?.task);
  const keys = useQuery({ queryKey: ["att", ds?.id, ds?.version], queryFn: () => api.attachments(ds!.id, ds!.version), enabled: !!ds });
  const [db, setDb] = useState("concert_singer");
  const schema = useQuery({ queryKey: ["ctx", ds?.id, ds?.version, db], queryFn: () => api.context(ds!.id, ds!.version, db), enabled: !!ds && !!db });
  const servable = (models.data ?? []).filter((m) => m.servable);
  const [picked, setPicked] = useState<string[]>(["gemma-4-e4b", "gemma-4-12b"]);
  const [question, setQuestion] = useState("How many singers are older than 30, by country?");
  const [confirmPaid, setConfirmPaid] = useState(false);
  const paid = picked.filter((id) => servable.find((m) => m.id === id)?.paid);
  const ask = useMutation({
    mutationFn: () => api.generate({ models: picked, dataset, inputs: { question, [task?.attachments?.key ?? "db_id"]: db }, confirm_paid: confirmPaid }),
  });
  const order = (id: string) => picked.indexOf(id);

  return (
    <div className="grid gap-6">
      <Card title="Playground">
        <div className="grid gap-4 lg:grid-cols-[1fr_1fr]">
          <div className="grid content-start gap-3">
            <div className="flex flex-wrap gap-3">
              <Field label="Dataset (for the task and databases)">
                <select className={inputCls} value={dataset} onChange={(e) => setDataset(e.target.value)}>
                  {withDbs.map((d) => <option key={d.path} value={d.path}>{d.id}@{d.version}</option>)}
                </select>
              </Field>
              <Field label={`Database (${keys.data?.keys.length ?? 0})`}>
                <select className={inputCls} value={db} onChange={(e) => setDb(e.target.value)}>
                  {(keys.data?.keys ?? []).map((k) => <option key={k} value={k}>{k}</option>)}
                </select>
              </Field>
            </div>
            <Field label="Question"><textarea dir="auto" rows={3} className={inputCls} value={question} onChange={(e) => setQuestion(e.target.value)} /></Field>
            <Field label="Models" hint="Local models answer through Ollama; export a fine-tuned run to use it here.">
              <div className="flex flex-wrap gap-2">
                {servable.map((m) => (
                  <label key={m.id} className={`flex items-center gap-1 rounded-md border px-2 py-1 text-sm ${picked.includes(m.id) ? "border-accent bg-accent-soft" : "border-line"}`}>
                    <input type="checkbox" checked={picked.includes(m.id)} onChange={() => setPicked(picked.includes(m.id) ? picked.filter((x) => x !== m.id) : [...picked, m.id])} />
                    {m.name}{m.paid && <Badge value="paid" />}
                  </label>
                ))}
              </div>
            </Field>
            {paid.length > 0 && <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={confirmPaid} onChange={(e) => setConfirmPaid(e.target.checked)} />Call the paid API ({paid.join(", ")})</label>}
            <div><Button disabled={!question || !db || picked.length === 0 || ask.isPending || (paid.length > 0 && !confirmPaid)} onClick={() => ask.mutate()}>{ask.isPending ? "Asking…" : "Ask"}</Button></div>
            <ErrorNote error={ask.error} />
          </div>
          <div className="min-w-0">
            <div className="font-mono text-xs uppercase text-muted">Schema in the prompt ({db})</div>
            <Code className="max-h-80 overflow-y-auto">{schema.data?.schema ?? (schema.error ? String(schema.error) : "…")}</Code>
          </div>
        </div>
      </Card>

      {ask.data && (
        <div className="grid gap-4 md:grid-cols-2">
          {ask.data.answers.map((a) => (
            <Card key={a.model} title={<span className="inline-flex items-center gap-2"><Swatch i={order(a.model)} />{a.name}</span>}
                  actions={a.latency_s !== undefined && <span className="num text-xs text-muted">{a.latency_s.toFixed(2)} s · {a.input_tokens ?? "?"} → {a.output_tokens ?? "?"} tokens</span>}>
              {a.error ? <p className="text-sm text-warn">{a.error}</p> : <>
                <Code>{a.sql ?? a.output}</Code>
                {a.result && <div className="mt-2"><ResultTable r={a.result} max={20} /></div>}
                {a.sql && a.output && a.output.trim() !== a.sql && <details className="mt-2"><summary className="cursor-pointer text-xs text-muted">raw output</summary><Code>{a.output}</Code></details>}
              </>}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
