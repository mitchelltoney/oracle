import { Panel } from "../components/Panel";
import { ProbBar } from "../components/ProbBar";
import { DISAGREEMENT_LOUD_THRESHOLD } from "../lib/consensus";
import { kickoffLabel, modelColor, pct } from "../lib/format";
import { selectConsensusRows } from "../lib/selectors";
import { useStore } from "../lib/store";

export function ConsensusView() {
  const rows = useStore(selectConsensusRows);
  const fixturesStatus = useStore((s) => s.endpoints.fixtures);

  if (rows.length === 0) {
    return (
      <div className="empty-state">
        {fixturesStatus.error !== null
          ? `fixtures endpoint unreachable // ${fixturesStatus.error}`
          : "no upcoming fixtures in the snapshot — run "}
        {fixturesStatus.error === null && <code>make ingest && make predict</code>}
      </div>
    );
  }

  return (
    <div className="consensus">
      {rows.map((row) => {
        const loud = !row.insufficient && row.jsdNorm >= DISAGREEMENT_LOUD_THRESHOLD;
        return (
          <Panel
            key={row.fixture.id}
            className={`consensus-row${loud ? " consensus-row--loud" : ""}`}
            title={`${row.fixture.home} vs ${row.fixture.away}`}
            sub={`${row.fixture.stage} // ${kickoffLabel(row.fixture.kickoff_utc)}`}
          >
            {row.models.length === 0 ? (
              <div className="empty-state">no predictions logged yet</div>
            ) : (
              <div className="consensus-row__models">
                {row.models.map((model) => (
                  <div key={model.model_version} className="consensus-model">
                    <span
                      className="consensus-model__tag"
                      style={{ color: modelColor(model.model_version) }}
                    >
                      {model.model_version}
                    </span>
                    <ProbBar probs={model.probs} />
                  </div>
                ))}
              </div>
            )}
            <div className="consensus-meter">
              <span>DISAGREEMENT (JSD)</span>
              <div className="consensus-meter__track">
                <div
                  className="consensus-meter__fill"
                  style={{ width: `${Math.min(1, row.jsdNorm) * 100}%` }}
                />
              </div>
              <span className="consensus-meter__value">
                {row.insufficient ? "insufficient models" : pct(row.jsdNorm)}
                {loud ? " // MODELS SPLIT" : ""}
              </span>
            </div>
          </Panel>
        );
      })}
    </div>
  );
}
