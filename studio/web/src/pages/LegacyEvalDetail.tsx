import { useQuery } from "@tanstack/react-query";
import { api, type LegacyComparison, type LegacyReport, type LegacyScore } from "../api";
import { Badge, Card } from "../ui";

// Mango evaluations (evaluate.py): a fine-tuned run vs its base on the mango suites.
const SUITE_LABEL: Record<string, string> = {
  "mango_mcq@v1": "Mango MCQ (accuracy)", "mango_ref@v1": "Open answers vs reference (chrF++)",
  "mango_advice@v1": "Mango advice (judge, 1–5)",
};
const SPLIT_LABEL: Record<string, string> = { heldout: "taught facts, new wording", heldout_unseen: "unseen documents" };

// MCQ accuracy and chrF++ are 0-1 and shown as percentages; the judge mean is on a 1-5 scale.
const isPct = (suite: string) => suite.startsWith("mango_mcq") || suite.startsWith("mango_ref");

function fmt(v: number, suite: string) {
  return isPct(suite) ? `${(v * 100).toFixed(1)}%` : v.toFixed(2);
}

export function LegacyEvalDetail({ id, r }: { id: string; r: LegacyReport }) {
  const examples = useQuery({ queryKey: ["examples", id], queryFn: () => api.examples(id) });
  const find = (m: string, c: LegacyComparison): LegacyScore | undefined =>
    r.scores.find((s) => s.model === m && s.suite === c.suite && s.slice.split === c.slice.split && s.slice.lang === c.slice.lang);

  return (
    <div className="grid gap-6">
      <div>
        <a href="#/evals" className="text-sm text-muted">← Evaluations</a>
        <h1 className="text-2xl font-bold">{id}</h1>
        <p className="text-sm text-muted">{r.dataset} · base vs <span className="font-mono">{r.models.at(-1)?.id}</span>
          {r.judge?.model ? ` · judge ${r.judge.model}, rubric ${r.judge.rubric}` : " · not judged (MCQ and chrF++ only)"}</p>
      </div>

      <Card title="Scorecard">
        <p className="mb-3 text-sm text-muted">Paired difference on the same items with a 95% bootstrap interval. A gain inside the noise shows as “no clear difference”.</p>
        <div className="overflow-x-auto">
          <table className="w-full min-w-[720px] text-sm">
            <thead className="text-left font-mono text-xs uppercase text-muted">
              <tr><th className="py-2">Suite</th><th>Split</th><th>Lang</th><th className="text-right">Base</th><th className="text-right">Tuned</th><th className="text-right">Δ [95% CI]</th><th className="text-right">n</th><th className="pl-4">Verdict</th></tr>
            </thead>
            <tbody>
              {r.comparisons.map((c, i) => {
                const a = find(c.a, c), b = find(c.b, c);
                const pct = isPct(c.suite);
                const d = (x: number) => (pct ? `${(x * 100).toFixed(1)}` : x.toFixed(2));
                return (
                  <tr key={i} className={`border-t border-line ${c.slice.lang === "all" ? "font-semibold" : ""}`}>
                    <td className="py-2">{SUITE_LABEL[c.suite] ?? c.suite}</td>
                    <td>{SPLIT_LABEL[c.slice.split] ?? c.slice.split}</td>
                    <td className="font-mono">{c.slice.lang}</td>
                    <td className="num text-right">{a ? fmt(a.value, c.suite) : "—"}</td>
                    <td className="num text-right">{b ? fmt(b.value, c.suite) : "—"}</td>
                    <td className="num text-right">{c.delta >= 0 ? "+" : ""}{d(c.delta)} [{d(c.ci95[0])}, {d(c.ci95[1])}]{pct ? " pp" : ""}</td>
                    <td className="num text-right text-muted">{b?.n ?? "—"}</td>
                    <td className="pl-4"><Badge value={c.verdict} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {(examples.data?.length ?? 0) > 0 && (
        <Card title="Biggest changes">
          <div className="grid gap-4">
            {examples.data!.slice(0, 12).map((ex) => (
              <div key={ex.item} className="rounded-md border border-line p-3">
                <div className="mb-2 flex justify-between gap-2 font-mono text-xs text-muted">
                  <span>{ex.item}</span>
                  <span className={ex.delta > 0 ? "text-accent" : ex.delta < 0 ? "text-warn" : ""}>Δ {ex.delta > 0 ? "+" : ""}{ex.delta}</span>
                </div>
                <div className="grid gap-3 md:grid-cols-2">
                  {(["base", "tuned"] as const).map((k) => (
                    <div key={k} className="min-w-0">
                      <div className="mb-1 font-mono text-xs uppercase text-muted">{k} · correctness {ex[k].verdict.correctness}/5</div>
                      <p dir="auto" className="whitespace-pre-wrap text-sm">{ex[k].answer}</p>
                      <p className="mt-1 text-xs italic text-muted">{ex[k].verdict.rationale}</p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  );
}
