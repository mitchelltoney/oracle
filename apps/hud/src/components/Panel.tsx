import type { CSSProperties, ReactNode } from "react";

export function Panel(props: {
  title: string;
  sub?: string;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
}) {
  return (
    <section
      className={`panel${props.className ? ` ${props.className}` : ""}`}
      style={props.style}
    >
      <header className="panel__header">
        {props.title}
        {props.sub !== undefined && <small>{props.sub}</small>}
      </header>
      <div className="panel__body">{props.children}</div>
    </section>
  );
}
