import { useRef, useState, type ReactNode, type UIEvent } from "react";

/**
 * Minimal fixed-row-height windowing — the feed can exceed 50 rows and the
 * performance rules require virtualization; a dependency is overkill for a
 * fixed-height terminal list.
 */
export function VirtualList<T>(props: {
  items: T[];
  rowHeight: number;
  overscan?: number;
  className?: string;
  renderRow: (item: T, index: number) => ReactNode;
}) {
  const { items, rowHeight, overscan = 8 } = props;
  const viewportRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [viewportHeight, setViewportHeight] = useState(600);

  const onScroll = (event: UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
    setViewportHeight(event.currentTarget.clientHeight);
  };

  const start = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const end = Math.min(
    items.length,
    Math.ceil((scrollTop + viewportHeight) / rowHeight) + overscan,
  );

  return (
    <div ref={viewportRef} className={props.className} onScroll={onScroll}>
      <div style={{ height: items.length * rowHeight, position: "relative" }}>
        {items.slice(start, end).map((item, offset) => (
          <div
            key={start + offset}
            style={{
              position: "absolute",
              top: (start + offset) * rowHeight,
              left: 0,
              right: 0,
              height: rowHeight,
            }}
          >
            {props.renderRow(item, start + offset)}
          </div>
        ))}
      </div>
    </div>
  );
}
