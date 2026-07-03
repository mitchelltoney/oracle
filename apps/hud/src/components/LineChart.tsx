/**
 * Hand-rolled SVG multi-series line chart. Data volumes are tiny (≤200
 * points per series), and owned markup keeps GSAP/glow styling possible
 * without fighting a chart library's animation system.
 */

export interface ChartSeries {
  label: string;
  color: string;
  points: { x: number; y: number }[];
}

const WIDTH = 320;
const HEIGHT = 150;
const PAD = { top: 8, right: 8, bottom: 18, left: 34 };

export function LineChart(props: {
  series: ChartSeries[];
  xTicks?: { x: number; label: string }[];
  yDomain?: [number, number];
}) {
  const [yMin, yMax] = props.yDomain ?? [0, 1];
  const xs = props.series.flatMap((s) => s.points.map((p) => p.x));
  const xMin = xs.length > 0 ? Math.min(...xs) : 0;
  const xMax = xs.length > 0 ? Math.max(...xs) : 1;
  const xSpan = xMax - xMin || 1;

  const sx = (x: number) =>
    PAD.left + ((x - xMin) / xSpan) * (WIDTH - PAD.left - PAD.right);
  const sy = (y: number) =>
    HEIGHT -
    PAD.bottom -
    ((y - yMin) / (yMax - yMin || 1)) * (HEIGHT - PAD.top - PAD.bottom);

  const yGrid = [0, 0.25, 0.5, 0.75, 1].filter((v) => v >= yMin && v <= yMax);

  return (
    <svg
      className="linechart"
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      width="100%"
      role="img"
    >
      {yGrid.map((v) => (
        <g key={v}>
          <line
            className="linechart__grid"
            x1={PAD.left}
            x2={WIDTH - PAD.right}
            y1={sy(v)}
            y2={sy(v)}
          />
          <text x={4} y={sy(v) + 3}>
            {Math.round(v * 100)}%
          </text>
        </g>
      ))}
      {(props.xTicks ?? []).map((tick) => (
        <text key={tick.x} x={sx(tick.x)} y={HEIGHT - 5} textAnchor="middle">
          {tick.label}
        </text>
      ))}
      {props.series.map((series) => (
        <g key={series.label}>
          <polyline
            fill="none"
            stroke={series.color}
            strokeWidth={1.5}
            points={series.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")}
          />
          {series.points.map((p, i) => (
            <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2} fill={series.color} />
          ))}
        </g>
      ))}
    </svg>
  );
}
