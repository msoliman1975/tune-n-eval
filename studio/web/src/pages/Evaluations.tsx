import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge, Card, ErrorNote, when } from "../ui";

export function Evaluations() {
  const evals = useQuery({ queryKey: ["evals"], queryFn: api.evals, refetchInterval: 5000 });
  return (
    <Card title="Evaluations" actions={<a className="rounded-md bg-accent px-3 py-1.5 text-sm text-bg" href="#/evals/new">New evaluation</a>}>
      <ErrorNote error={evals.error} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[760px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted">
            <tr><th className="py-2">Evaluation</th><th>Status</th><th>Dataset</th><th>Models and headline score</th><th>Created</th></tr>
          </thead>
          <tbody>
            {(evals.data ?? []).map((e) => (
              <tr key={e.id} className="border-t border-line align-top">
                <td className="py-2"><a className="text-accent" href={`#/evals/${e.id}`}>{e.id}</a>{e.kind === "mango" && <div className="text-xs text-muted">mango suites</div>}</td>
                <td><Badge value={e.status} /></td>
                <td className="font-mono text-xs">{e.dataset ?? "—"}{e.n_items ? <div className="text-muted">{e.n_items} items</div> : null}</td>
                <td className="text-xs">
                  {(e.models ?? []).map((m) => (
                    <div key={m} className="flex gap-2"><span className="font-mono">{m}</span>
                      {e.headline?.[m] !== undefined && <span className="num text-muted">{e.primary_metric} {(e.headline[m] * 100).toFixed(1)}%</span>}</div>
                  ))}
                </td>
                <td className="text-muted">{when(e.created_at)}</td>
              </tr>
            ))}
            {evals.data?.length === 0 && <tr><td colSpan={5} className="py-6 text-center text-muted">No evaluations yet.</td></tr>}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
