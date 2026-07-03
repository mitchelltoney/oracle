import gsap from "gsap";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { startPoller } from "./lib/poller";
import { store, useStore } from "./lib/store";
import type { EndpointName } from "./lib/types";
import { BracketView } from "./views/BracketView";
import { ConsensusView } from "./views/ConsensusView";
import { FeedView } from "./views/FeedView";
import { ReportCardView } from "./views/ReportCardView";
import { TimelineView } from "./views/TimelineView";

const VIEWS = [
  { id: "bracket", label: "Bracket", component: BracketView },
  { id: "consensus", label: "Consensus", component: ConsensusView },
  { id: "report", label: "Report Card", component: ReportCardView },
  { id: "timeline", label: "Timeline", component: TimelineView },
  { id: "feed", label: "Event Feed", component: FeedView },
] as const;

type ViewId = (typeof VIEWS)[number]["id"];

const ENDPOINTS: EndpointName[] = ["fixtures", "predictions", "calibration", "sim"];

function StatusBar() {
  const endpoints = useStore((s) => s.endpoints);
  const lastTickAt = useStore((s) => s.lastTickAt);
  const sim = useStore((s) => s.sim);
  return (
    <div className="statusbar">
      {ENDPOINTS.map((name) => {
        const status = endpoints[name];
        const dot = !status.ok
          ? "down"
          : status.lastOkAt === null
            ? "empty"
            : "ok";
        return (
          <span key={name} className="statusbar__endpoint" title={status.error ?? ""}>
            <span className={`statusbar__dot statusbar__dot--${dot}`} />/{name}
          </span>
        );
      })}
      <span className="masthead__spacer" />
      {sim !== null && (
        <span>
          sim {sim.model_version} // {sim.n_sims.toLocaleString()} runs //{" "}
          {sim.generated_at.slice(0, 16)}Z
        </span>
      )}
      <span>
        last poll{" "}
        {lastTickAt === null ? "—" : new Date(lastTickAt).toISOString().slice(11, 19)}
      </span>
    </div>
  );
}

export function App() {
  const [view, setView] = useState<ViewId>("bracket");
  const mainRef = useRef<HTMLElement>(null);

  useEffect(() => startPoller(store), []);

  // Panel transition on view switch; tween killed on effect cleanup.
  useLayoutEffect(() => {
    const node = mainRef.current;
    if (!node) return;
    const tween = gsap.fromTo(
      node,
      { opacity: 0, y: 8 },
      { opacity: 1, y: 0, duration: 0.35, ease: "power2.out" },
    );
    return () => {
      tween.kill();
    };
  }, [view]);

  const Active =
    VIEWS.find((v) => v.id === view)?.component ?? BracketView;

  return (
    <div className="app">
      <header className="masthead">
        <h1 className="masthead__title">WC ORACLE</h1>
        <span className="masthead__sub">WORLD CUP 2026 // KNOCKOUT STAGE</span>
        <span className="masthead__spacer" />
        <nav className="nav">
          {VIEWS.map((v) => (
            <button
              key={v.id}
              className={`nav__tab${v.id === view ? " nav__tab--active" : ""}`}
              onClick={() => setView(v.id)}
            >
              {v.label}
            </button>
          ))}
        </nav>
      </header>
      <StatusBar />
      <main ref={mainRef} className="main">
        <Active />
      </main>
    </div>
  );
}
