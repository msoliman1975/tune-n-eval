import type { ReactNode } from "react";

export function Card({ title, children, actions }: { title?: string; children: ReactNode; actions?: ReactNode }) {
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
  b_better: "bg-accent-soft text-accent",
  running: "bg-mango-soft text-mango", started: "bg-mango-soft text-mango", queued: "bg-mango-soft text-mango",
  no_clear_difference: "bg-bg text-muted",
  failed: "bg-warn-soft text-warn", error: "bg-warn-soft text-warn", a_better: "bg-warn-soft text-warn",
  low_text: "bg-warn-soft text-warn", cancelled: "bg-bg text-muted",
};
const LABELS: Record<string, string> = { b_better: "tuned better", a_better: "base better", no_clear_difference: "no clear difference" };

export function Badge({ value }: { value: string | null | undefined }) {
  const v = value ?? "unknown";
  return (
    <span className={`inline-block rounded px-2 py-0.5 font-mono text-xs ${TONES[v] ?? "bg-bg text-muted"}`}>
      {LABELS[v] ?? v.replace(/_/g, " ")}
    </span>
  );
}

export function Button(props: React.ButtonHTMLAttributes<HTMLButtonElement> & { tone?: "primary" | "plain" }) {
  const { tone = "primary", className = "", ...rest } = props;
  const look = tone === "primary"
    ? "bg-accent text-bg hover:opacity-90 disabled:opacity-40"
    : "border border-line bg-surface hover:bg-accent-soft disabled:opacity-40";
  return <button {...rest} className={`rounded-md px-3 py-1.5 text-sm font-medium ${look} ${className}`} />;
}

export function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-1 text-sm">
      <span className="font-mono text-xs uppercase tracking-wide text-muted">{label}</span>
      {children}
    </label>
  );
}

export const inputCls = "rounded-md border border-line bg-bg px-2 py-1.5 text-sm";

export function ErrorNote({ error }: { error: unknown }) {
  if (!error) return null;
  return <p className="rounded-md bg-warn-soft px-3 py-2 text-sm text-warn">{String((error as Error).message ?? error)}</p>;
}

export function when(ts: string | null | undefined) {
  return ts ? new Date(ts).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—";
}
