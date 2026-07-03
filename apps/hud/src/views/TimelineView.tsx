import { useState } from "react";

import { LineChart, type ChartSeries } from "../components/LineChart";
import { Panel } from "../components/Panel";
import { modelColor } from "../lib/format";
import {
  selectTimelineFixtures,
  selectTimelineSeries,
  type TimelineSeries,
} from "../lib/selectors";
import { useStore } from "../lib/store";
import type { TimelinePoint } from "../lib/types";

const EMPTY_SERIES: TimelineSeries[] = [];

const OUTCOMES: { key: keyof Pick<TimelinePoint, "h" | "d" | "a">; label: string }[] = [
  { key: "h", label: "P(HOME WIN)" },
  { key: "d", label: "P(DRAW)" },
  { key: "a", label: "P(AWAY WIN)" },
];

export function TimelineView() {
  const options = useStore(selectTimelineFixtures);
  const [selected, setSelected] = useState<number | null>(null);
  // Derive the active fixture: a stale selection (fixture no longer in the
  // data) falls back to the first option — no state sync needed.
  const activeId =
    selected !== null && options.some((o) => o.fixtureId === selected)
      ? selected
      : (options[0]?.fixtureId ?? null);
  // Stable empty reference: useSyncExternalStore snapshots must be
  // referentially stable, a fresh [] per call would loop.
  const series = useStore((s) =>
    activeId === null ? EMPTY_SERIES : selectTimelineSeries(s, activeId),
  );

  if (options.length === 0) {
    return (
      <div className="empty-state">
        no predictions logged yet — run <code>make predict</code>
      </div>
    );
  }

  const active = options.find((o) => o.fixtureId === activeId);
  const totalPoints = series.reduce((acc, s) => acc + s.points.length, 0);

  const chartFor = (key: "h" | "d" | "a"): ChartSeries[] =>
    series.map((s) => ({
      label: s.modelVersion,
      color: cssColor(s.modelVersion),
      points: s.points.map((p) => ({ x: new Date(p.t).getTime(), y: p[key] })),
    }));

  const xTicks = (() => {
    const ts = [
      ...new Set(series.flatMap((s) => s.points.map((p) => p.t.slice(0, 10)))),
    ].sort();
    return ts.map((day) => ({
      x: new Date(`${day}T12:00:00Z`).getTime(),
      label: day.slice(5),
    }));
  })();

  return (
    <div>
      <div className="timeline__picker">
        <span>FIXTURE</span>
        <select
          value={activeId ?? ""}
          onChange={(e) => setSelected(Number(e.target.value))}
        >
          {options.map((option) => (
            <option key={option.fixtureId} value={option.fixtureId}>
              {option.label} ({option.pointCount} pts)
            </option>
          ))}
        </select>
        <span>
          history accumulates from live polling — {totalPoints} points so far
          {/* full history needs the /predictions history endpoint (TODO in api.ts) */}
        </span>
      </div>
      <div className="timeline-charts">
        {OUTCOMES.map((outcome) => (
          <Panel
            key={outcome.key}
            title={outcome.label}
            sub={active?.label ?? ""}
          >
            <LineChart series={chartFor(outcome.key)} xTicks={xTicks} />
          </Panel>
        ))}
      </div>
      <div className="timeline-legend">
        {series.map((s) => (
          <span key={s.modelVersion}>
            <span
              className="timeline-legend__swatch"
              style={{ background: cssColor(s.modelVersion) }}
            />
            {s.modelVersion}
          </span>
        ))}
      </div>
    </div>
  );
}

/** SVG stroke needs a concrete value; resolve the CSS variable by family. */
function cssColor(modelVersion: string): string {
  const varName = modelColor(modelVersion);
  const fallback: Record<string, string> = {
    "var(--model-dc)": "#40dcff",
    "var(--model-elo)": "#b487ff",
    "var(--model-gbm)": "#ffb340",
    "var(--model-ens)": "#2ee6c8",
    "var(--accent)": "#40dcff",
  };
  return fallback[varName] ?? "#40dcff";
}
