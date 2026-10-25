import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "./api";
import { DataPage } from "./pages/Data";
import { EvalDetail } from "./pages/EvalDetail";
import { Evaluations } from "./pages/Evaluations";
import { RunDetail } from "./pages/RunDetail";
import { Training } from "./pages/Training";

// Hash routes: #/training, #/runs/<id>, #/evals, #/evals/<id>, #/data
function useRoute() {
  const [hash, setHash] = useState(location.hash || "#/training");
  useEffect(() => {
    const on = () => setHash(location.hash || "#/training");
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  return hash.slice(2).split("/");
}

const NAV = [["training", "Training"], ["evals", "Evaluations"], ["data", "Data"]] as const;

export function App() {
  const [page, id] = useRoute();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const section = page === "runs" ? "training" : page;

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-baseline gap-6">
          <a href="#/training" className="font-display text-2xl font-bold text-fg no-underline">AgriTune <span className="text-mango">Studio</span></a>
          <nav className="flex gap-1">
            {NAV.map(([key, label]) => (
              <a key={key} href={`#/${key}`}
                 className={`rounded-md px-3 py-1.5 text-sm ${section === key ? "bg-accent text-bg" : "text-fg hover:bg-accent-soft"}`}>
                {label}
              </a>
            ))}
          </nav>
        </div>
        <div className="flex gap-3 font-mono text-xs text-muted" aria-label="Service status">
          {Object.entries(health.data ?? {}).map(([svc, up]) => (
            <span key={svc} className="flex items-center gap-1">
              <span className={`inline-block h-2 w-2 rounded-full ${up ? "bg-accent" : "bg-warn"}`} />{svc}
            </span>
          ))}
        </div>
      </header>
      {page === "runs" && id ? <RunDetail id={id} />
        : page === "evals" && id ? <EvalDetail id={id} />
        : page === "evals" ? <Evaluations />
        : page === "data" ? <DataPage />
        : <Training />}
    </div>
  );
}
