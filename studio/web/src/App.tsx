import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { api } from "./api";
import { DatasetExplorer } from "./pages/DatasetExplorer";
import { Datasets } from "./pages/Datasets";
import { DraftDetail } from "./pages/DraftDetail";
import { EvalDetail } from "./pages/EvalDetail";
import { EvalNew } from "./pages/EvalNew";
import { Evaluations } from "./pages/Evaluations";
import { ModelDetail, Models } from "./pages/Models";
import { Playground } from "./pages/Playground";
import { RunDetail } from "./pages/RunDetail";
import { Training } from "./pages/Training";

// Hash routes: #/datasets, #/datasets/<id>/<v>?tab=overview|data|columns|databases|task|usage&db=<key>, #/drafts/<id>, #/models, #/models/<id>, #/training, #/runs/<id>,
// #/evals, #/evals/new?models=a,b&dataset=..., #/evals/<id>, #/playground
function useRoute() {
  const [hash, setHash] = useState(location.hash || "#/datasets");
  useEffect(() => {
    const on = () => setHash(location.hash || "#/datasets");
    addEventListener("hashchange", on);
    return () => removeEventListener("hashchange", on);
  }, []);
  const [path, query] = hash.slice(2).split("?");
  return { parts: path.split("/").map(decodeURIComponent), query: new URLSearchParams(query ?? "") };
}

const NAV = [["datasets", "Datasets"], ["models", "Models"], ["training", "Training"], ["evals", "Evaluations"], ["playground", "Playground"]] as const;
const SECTION: Record<string, string> = { drafts: "datasets", runs: "training" };

export function App() {
  const { parts: [page, a, b], query } = useRoute();
  const health = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 15000 });
  const section = SECTION[page] ?? page;

  return (
    <div className="mx-auto max-w-6xl px-4 pb-16 pt-6">
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-wrap items-baseline gap-6">
          <a href="#/datasets" className="font-display text-2xl font-bold text-fg no-underline">AgriTune <span className="text-mango">Studio</span></a>
          <nav className="flex flex-wrap gap-1">
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
      {page === "datasets" && a && b ? <DatasetExplorer id={a} version={b} query={query} />
        : page === "drafts" && a ? <DraftDetail id={a} />
        : page === "models" && a ? <ModelDetail id={a} />
        : page === "models" ? <Models />
        : page === "training" ? <Training />
        : page === "runs" && a ? <RunDetail id={a} />
        : page === "evals" && a === "new" ? <EvalNew query={query} />
        : page === "evals" && a ? <EvalDetail id={a} />
        : page === "evals" ? <Evaluations />
        : page === "playground" ? <Playground />
        : <Datasets />}
    </div>
  );
}
