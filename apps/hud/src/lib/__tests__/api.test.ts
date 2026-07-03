import { afterEach, describe, expect, it, vi } from "vitest";

import {
  fetchCalibration,
  fetchFixtures,
  fetchPredictionHistory,
  fetchPredictions,
  fetchSim,
} from "../api";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("hits /api paths and passes typed payloads through on 200", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([{ id: 1 }]));
    vi.stubGlobal("fetch", fetchMock);
    const result = await fetchFixtures();
    expect(fetchMock).toHaveBeenCalledWith("/api/fixtures", expect.anything());
    expect(result).toEqual({ kind: "ok", data: [{ id: 1 }] });
  });

  it("appends model_version only when provided", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([]));
    vi.stubGlobal("fetch", fetchMock);
    await fetchPredictions("dc-1.0.0");
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/predictions?model_version=dc-1.0.0",
      expect.anything(),
    );
    await fetchPredictions();
    expect(fetchMock).toHaveBeenLastCalledWith(
      "/api/predictions",
      expect.anything(),
    );
  });

  it("maps 404 to empty (data not generated yet, not an error)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ detail: "no snapshot" }, 404)),
    );
    expect(await fetchFixtures()).toEqual({ kind: "empty" });
    expect(await fetchSim()).toEqual({ kind: "empty" });
    expect(await fetchCalibration()).toEqual({ kind: "empty" });
  });

  it("maps HTTP errors to error results without throwing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(jsonResponse({}, 500)));
    const result = await fetchSim();
    expect(result.kind).toBe("error");
    if (result.kind === "error") expect(result.message).toContain("500");
  });

  it("maps network failures to error results without throwing", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));
    const result = await fetchPredictions();
    expect(result).toEqual({ kind: "error", message: "ECONNREFUSED" });
  });

  it("maps malformed JSON bodies to error results", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response("<html>oops</html>", { status: 200 })),
    );
    const result = await fetchCalibration();
    expect(result.kind).toBe("error");
  });

  it("prediction history is stubbed empty until the API grows an endpoint", async () => {
    // Pins the TODO(api) contract in api.ts — no fetch is made.
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    expect(await fetchPredictionHistory(537420)).toEqual({ kind: "empty" });
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
