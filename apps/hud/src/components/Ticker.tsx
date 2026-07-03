import gsap from "gsap";
import { useLayoutEffect, useRef } from "react";

/**
 * Animated number readout: tweens between values on change instead of
 * snapping. Text content is owned entirely by the effect (GSAP writes it
 * imperatively); the tween is killed on unmount and on re-target — no
 * leaked animations (hud-engineer rule).
 */
export function Ticker(props: {
  value: number;
  format: (value: number) => string;
  className?: string;
}) {
  const nodeRef = useRef<HTMLSpanElement>(null);
  const shownRef = useRef<number | null>(null);
  const { value, format } = props;

  useLayoutEffect(() => {
    const node = nodeRef.current;
    if (!node) return;
    if (shownRef.current === null) {
      shownRef.current = value;
      node.textContent = format(value);
      return;
    }
    const state = { v: shownRef.current };
    const tween = gsap.to(state, {
      v: value,
      duration: 0.6,
      ease: "power2.out",
      onUpdate: () => {
        shownRef.current = state.v;
        node.textContent = format(state.v);
      },
    });
    return () => {
      tween.kill();
    };
  }, [value, format]);

  return <span ref={nodeRef} className={props.className} />;
}
