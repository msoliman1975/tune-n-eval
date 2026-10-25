import { useQuery } from "@tanstack/react-query";
import { api } from "../api";
import { Badge, Card, ErrorNote, when } from "../ui";

export function Evaluations() {
  const evals = useQuery({ queryKey: ["evals"], queryFn: api.evals, refetchInterval: 5000 });
  return (
    <Card title="Evaluations">
      <ErrorNote error={evals.error} />
      <div className="overflow-x-auto">
        <table className="w-full min-w-[640px] text-sm">
          <thead className="text-left font-mono text-xs uppercase text-muted">
            <tr><th className="py-2">Evaluation</th><th>Status</th><th>Dataset</th><th>Fine-tuned model</th><th>Created</th></tr>
          </thead>
          <tbody>
            {(evals.data ?? []).map((e) => (
              <tr key={e.id} className="border-t border-line">
                <td className="py-2"><a className="text-accent" href={`#/evals/${e.id}`}>{e.id}</a></td>
                <td><Badge value={e.status} /></td>
                <td>{e.dataset ?? "—"}</td>
                <td className="font-mono text-xs">{e.models?.at(-1) ?? "—"}</td>
                <td className="text-muted">{when(e.created_at)}</td>
              </tr>
            ))}
            {evals.data?.length === 0 && <tr><td colSpan={5} className="py-6 text-center text-muted">No evaluations yet. Start one from a finished run.</td></tr>}
          </tbody>
        </table>
      </div>
    </Card>
  );
}
