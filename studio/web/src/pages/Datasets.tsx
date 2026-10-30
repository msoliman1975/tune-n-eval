import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { api } from "../api";
import { Badge, Button, Card, ErrorNote, Field, inputCls, when } from "../ui";

export function Datasets() {
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const drafts = useQuery({ queryKey: ["drafts"], queryFn: api.drafts, refetchInterval: 10000 });
  const tasks = useQuery({ queryKey: ["tasks"], queryFn: api.tasks });
  const taskName = (ref?: string) => tasks.data?.find((t) => t.ref === ref)?.name ?? ref ?? "—";
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const upload = useMutation({ mutationFn: () => api.upload(file!, name), onSuccess: (d) => { location.hash = `#/drafts/${d.id}`; } });
  const open = (drafts.data ?? []).filter((d) => d.status !== "frozen");

  return (
    <div className="grid gap-6">
      <Card title="New dataset from a file">
        <p className="mb-3 text-sm text-muted">
          Upload JSONL, JSON, CSV or Parquet. Next you map its columns to a task (e.g. NL→SQL: question, database, gold SQL),
          attach files the task needs (SQLite databases), check and edit items, and freeze an immutable version.
        </p>
        <div className="flex flex-wrap items-end gap-4">
          <Field label="File"><input type="file" accept=".jsonl,.json,.csv,.parquet" className="text-sm" onChange={(e) => setFile(e.target.files?.[0] ?? null)} /></Field>
          <Field label="Name"><input className={inputCls} value={name} placeholder={file?.name.replace(/\.\w+$/, "") ?? "my dataset"} onChange={(e) => setName(e.target.value)} /></Field>
          <Button disabled={!file || upload.isPending} onClick={() => upload.mutate()}>{upload.isPending ? "Uploading…" : "Upload"}</Button>
        </div>
        <ErrorNote error={upload.error} />
      </Card>

      {open.length > 0 && (
        <Card title="Drafts">
          <Table head={["Draft", "Status", "Task", "Items", "Updated"]}>
            {open.map((d) => (
              <tr key={d.id} className="border-t border-line">
                <td className="py-2"><a className="text-accent" href={`#/drafts/${d.id}`}>{d.name}</a><div className="font-mono text-xs text-muted">{d.filename}</div></td>
                <td><Badge value={d.status} /></td>
                <td>{taskName(d.task)}</td>
                <td className="num">{Object.entries(d.counts ?? {}).filter(([, n]) => n).map(([k, n]) => `${k} ${n}`).join(" · ") || `${d.n_rows} rows`}</td>
                <td className="text-muted">{when(d.updated_at ?? d.created_at)}</td>
              </tr>
            ))}
          </Table>
        </Card>
      )}

      <Card title="Frozen datasets">
        <ErrorNote error={datasets.error} />
        <Table head={["Dataset", "Task", "Splits", "Split by", "Created"]}>
          {(datasets.data ?? []).map((d) => (
            <tr key={d.path} className="border-t border-line">
              <td className="py-2"><a className="font-mono text-accent" href={`#/datasets/${d.id}/${d.version}`}>{d.id}@{d.version}</a></td>
              <td>{taskName(d.task)}</td>
              <td className="num">{Object.entries(d.counts).map(([k, v]) => `${k} ${v}`).join(" · ")}</td>
              <td className="font-mono text-xs text-muted">{d.split_by}</td>
              <td className="text-muted">{when(d.created_at)}</td>
            </tr>
          ))}
        </Table>
      </Card>

      <details className="rounded-lg border border-line bg-surface p-5">
        <summary className="cursor-pointer text-lg font-bold">Mango document pipeline</summary>
        <div className="mt-4"><MangoPipeline /></div>
      </details>
    </div>
  );
}

function Table({ head, children }: { head: string[]; children: React.ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] text-sm">
        <thead className="text-left font-mono text-xs uppercase text-muted"><tr>{head.map((h) => <th key={h} className="py-2 pr-3">{h}</th>)}</tr></thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

function MangoPipeline() {
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });
  const pipeline = useQuery({ queryKey: ["pipeline"], queryFn: api.pipeline, refetchInterval: 30000 });
  const p = pipeline.data;
  return (
    <div className="grid gap-4">
      <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
        {[["Chunks", p?.chunks], ["Synthesized", p?.chunks_synthesized], ["Items", p?.items], ["Dropped by checks", p?.dropped]].map(([k, v]) => (
          <div key={k as string}><dt className="font-mono text-xs uppercase text-muted">{k}</dt><dd className="num text-2xl">{v ?? "—"}</dd></div>
        ))}
      </dl>
      <p className="text-sm text-muted">Stages run from the CLI: <code className="font-mono">bash services/data/run.sh collect | parse | chunk | synth | freeze</code></p>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted">
            <tr><th className="py-2">Document</th><th>Lang</th><th>Egypt</th><th>License</th><th>Download</th><th>Parse</th><th className="text-right">Chars</th></tr>
          </thead>
          <tbody>
            {(sources.data ?? []).map((s) => (
              <tr key={s.doc_id} className="border-t border-line align-top">
                <td className="py-2 pr-3"><div dir="auto">{s.title}</div><div className="font-mono text-xs text-muted">{s.doc_id}</div></td>
                <td className="font-mono">{s.lang}</td>
                <td>{s.egypt_specific ? "yes" : ""}</td>
                <td className="max-w-48 text-xs text-muted">{s.license}</td>
                <td><Badge value={s.download} /></td>
                <td><Badge value={s.parse} /></td>
                <td className="num text-right">{s.chars?.toLocaleString() ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
