import type { ReactNode } from "react";
import type { ExecPreview } from "./api";

export function Card({ title, children, actions }: { title?: ReactNode; children: ReactNode; actions?: ReactNode }) {
  return (
    <section className="rounded-lg border border-line bg-surface p-5 min-w-0">
      {(title || actions) && (
        <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2">
          {title && <h2 className="text-lg font-bold">{title}</h2>}
          {actions}
        </div>
      )}
      {children}
    </section>
  );
}

const TONES: Record<string, string> = {
  succeeded: "bg-accent-soft text-accent", ok: "bg-accent-soft text-accent", cached: "bg-accent-soft text-accent",
  b_better: "bg-accent-soft text-accent", frozen: "bg-accent-soft text-accent", pass: "bg-accent-soft text-accent",
  running: "bg-mango-soft text-mango", started: "bg-mango-soft text-mango", queued: "bg-mango-soft text-mango",
  freezing: "bg-mango-soft text-mango", mapped: "bg-mango-soft text-mango", uploaded: "bg-mango-soft text-mango",
  paid: "bg-mango-soft text-mango",
  no_clear_difference: "bg-bg text-muted",
  failed: "bg-warn-soft text-warn", error: "bg-warn-soft text-warn", a_better: "bg-warn-soft text-warn", fail: "bg-warn-soft text-warn",
  low_text: "bg-warn-soft text-warn", cancelled: "bg-bg text-muted", freeze_failed: "bg-warn-soft text-warn",
};
const LABELS: Record<string, string> = { b_better: "tuned better", a_better: "base better", no_clear_difference: "no clear difference" };

export function Badge({ value, label }: { value: string | null | undefined; label?: string }) {
  const v = value ?? "unknown";
  return (
    <span className={`inline-block rounded px-2 py-0.5 font-mono text-xs ${TONES[v] ?? "bg-bg text-muted"}`}>
      {label ?? LABELS[v] ?? v.replace(/_/g, " ")}
    </span>
  );
}

export function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: "primary" | "plain" | "danger" }) {
  const { tone = "primary", className = "", ...rest } = props;
  const look = tone === "primary"
    ? "bg-accent text-bg hover:opacity-90 disabled:opacity-40"
    : tone === "danger" ? "border border-warn text-warn hover:bg-warn-soft disabled:opacity-40"
    : "border border-line bg-surface hover:bg-accent-soft disabled:opacity-40";
  return <button {...rest} className={`rounded-md px-3 py-1.5 text-sm font-medium ${look} ${className}`} />;
}

export function Field({ label, children, hint }: { label: string; children: ReactNode; hint?: ReactNode }) {
  return (
    <label className="grid gap-1 text-sm">
      <span className="font-mono text-xs uppercase tracking-wide text-muted">{label}</span>
      {children}
      {hint && <span className="text-xs text-muted">{hint}</span>}
    </label>
  );
}

export const inputCls = "rounded-md border border-line bg-bg px-2 py-1.5 text-sm min-w-0";

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return <p className="rounded-md bg-warn-soft px-3 py-2 text-sm text-warn">{String((error as Error).message ?? error)}</p>;
}

export function when(ts: string | null | undefined) {
  return ts ? new Date(ts).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";
}

/** Model i's series color (fixed order, never cycled past 8). */
export const seriesVar = (i: number) => `var(--color-series-${Math.min(i, 7) + 1})`;

export function Swatch({ i }: { i: number }) {
  return <span aria-hidden className="inline-block h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: seriesVar(i) }} />;
}

export function Code({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <pre className={`overflow-x-auto whitespace-pre-wrap break-words rounded-md bg-bg p-3 font-mono text-xs ${className}`}>{children}</pre>;
}

export function ResultTable({ r, max = 10 }: { r?: ExecPreview | null; max?: number }) {
  if (!r) return null;
  if (r.error) return <p className="font-mono text-xs text-warn">{r.error}</p>;
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full font-mono text-xs">
        <thead className="text-left text-muted"><tr>{r.columns.map((c, i) => <th key={i} className="pr-3 font-medium">{c}</th>)}</tr></thead>
        <tbody>
          {r.rows.slice(0, max).map((row, i) => (
            <tr key={i} className="border-t border-line">{row.map((v, j) => <td key={j} className="pr-3 num">{v === null ? "NULL" : String(v)}</td>)}</tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-xs text-muted">{r.n_rows}{r.truncated ? "+" : ""} row{r.n_rows === 1 ? "" : "s"}{r.n_rows > max ? `, first ${Math.min(max, r.rows.length)} shown` : ""}</p>
    </div>
  );
}

export function Pager({ total, offset, limit, onChange }: { total: number; offset: number; limit: number; onChange: (o: number) => void }) {
  if (total <= limit) return <p className="text-xs text-muted">{total} item{total === 1 ? "" : "s"}</p>;
  return (
    <div className="flex items-center gap-2 text-sm">
      <Button tone="plain" disabled={offset === 0} onClick={() => onChange(Math.max(0, offset - limit))}>← Prev</Button>
      <span className="num text-muted">{offset + 1}–{Math.min(total, offset + limit)} of {total}</span>
      <Button tone="plain" disabled={offset + limit >= total} onClick={() => onChange(offset + limit)}>Next →</Button>
    </div>
  );
}

export function Tabs<T extends string>({ value, options, onChange }: { value: T; options: [T, string][]; onChange: (v: T) => void }) {
  return (
    <div className="flex flex-wrap gap-1" role="tablist">
      {options.map(([k, label]) => (
        <button key={k} role="tab" aria-selected={value === k} onClick={() => onChange(k)}
                className={`rounded-md px-2.5 py-1 text-sm ${value === k ? "bg-accent text-bg" : "border border-line hover:bg-accent-soft"}`}>
          {label}
        </button>
      ))}
    </div>
  );
}
