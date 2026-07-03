import { Panel } from "../components/Panel";
import { Ticker } from "../components/Ticker";
import { modelColor } from "../lib/format";
import { selectCalibrationRanked } from "../lib/selectors";
import { useStore } from "../lib/store";

const fixed4 = (value: number) => value.toFixed(4);

export function ReportCardView() {
  const rows = useStore(selectCalibrationRanked);
  const status = useStore((s) => s.endpoints.calibration);

  if (rows.length === 0) {
    return (
      <div className="empty-state">
        {status.error !== null
          ? `calibration endpoint unreachable // ${status.error}`
          : "no scored predictions yet — calibration needs finished fixtures in the snapshot"}
      </div>
    );
  }

  return (
    <div className="reportcard">
      {rows.map((row) => (
        <Panel
          key={row.model_version}
          title={row.model_version}
          sub={`rank ${row.rank} of ${rows.length}`}
          style={{ borderTop: `2px solid ${modelColor(row.model_version)}` }}
        >
          <div className="reportcard-model__rank">
            #{row.rank}
          </div>
          <div className="reportcard-model__grid">
            <div>
              <div className="reportcard-metric__label">Brier</div>
              <Ticker
                className="reportcard-metric__value"
                value={row.brier}
                format={fixed4}
              />
            </div>
            <div>
              <div className="reportcard-metric__label">Log loss</div>
              <Ticker
                className="reportcard-metric__value"
                value={row.log_loss}
                format={fixed4}
              />
            </div>
            <div>
              <div className="reportcard-metric__label">Scored (n)</div>
              <div className="reportcard-metric__value">{row.n}</div>
            </div>
          </div>
        </Panel>
      ))}
    </div>
  );
}
