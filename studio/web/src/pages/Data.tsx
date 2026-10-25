import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge, Card, ErrorNote, when } from "../ui";

export function DataPage() {
  const sources = useQuery({ queryKey: ["sources"], queryFn: api.sources });
  const pipeline = useQuery({ queryKey: ["pipeline"], queryFn: api.pipeline, refetchInterval: 10000 });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const p = pipeline.data;

  return (
    <div className="grid gap-6">
      <Card title="Pipeline">
        <ErrorNote error={pipeline.error} />
        <dl className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
          {[["Chunks", p?.chunks], ["Synthesized", p?.chunks_synthesized], ["Items", p?.items], ["Dropped by checks", p?.dropped]].map(([k, v]) => (
            <div key={k as string}><dt className="font-mono text-xs uppercase text-muted">{k}</dt><dd className="num text-2xl">{v ?? "—"}</dd></div>
          ))}
        </dl>
        <p className="mt-3 text-sm text-muted">Stages run from the CLI: <code className="font-mono">bash services/data/run.sh collect | parse | chunk | synth | freeze</code></p>
      </Card>

      <Card title="Frozen datasets">
        <ul className="grid gap-2 text-sm">
          {(datasets.data ?? []).map((d) => (
            <li key={d.path} className="flex flex-wrap justify-between gap-2 border-t border-line pt-2">
              <span className="font-mono">{d.id}@{d.version}</span>
              <span className="num text-muted">{Object.entries(d.counts).map(([k, v]) => `${k} ${v}`).join(" · ")}</span>
              <span className="text-muted">{when(d.created_at)}</span>
            </li>
          ))}
        </ul>
      </Card>

      <Card title={`Sources (${sources.data?.length ?? 0})`}>
        <ErrorNote error={sources.error} />
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
      </Card>
    </div>
  );
}
