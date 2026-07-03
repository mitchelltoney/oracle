import { VirtualList } from "../components/VirtualList";
import { clockLabel } from "../lib/format";
import { useStore } from "../lib/store";
import type { FeedEvent } from "../lib/types";

const ROW_HEIGHT = 20;

function FeedLine(props: { event: FeedEvent }) {
  const { event } = props;
  return (
    <div className={`feed-line feed-line--${event.severity}`}>
      <span className="feed-line__time">{clockLabel(event.at)}</span>
      <span className="feed-line__kind">[{event.kind}]</span>
      <span className="feed-line__text">{event.text}</span>
    </div>
  );
}

export function FeedView() {
  const feed = useStore((s) => s.feed);

  return (
    <div className="feed">
      {feed.length === 0 ? (
        <div className="empty-state">awaiting first poll tick…</div>
      ) : (
        <VirtualList
          className="feed__viewport"
          items={feed}
          rowHeight={ROW_HEIGHT}
          renderRow={(event) => <FeedLine event={event} />}
        />
      )}
    </div>
  );
}
