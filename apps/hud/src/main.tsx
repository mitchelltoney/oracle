import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { store } from "./lib/store";
import { loadTimeline } from "./lib/timeline";
import "./styles/global.css";

// Hydrate accumulated prediction history before the first poll tick.
store.commitTick({ ...store.getState(), timeline: loadTimeline(localStorage) });

const container = document.getElementById("root");
if (!container) throw new Error("missing #root element");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
