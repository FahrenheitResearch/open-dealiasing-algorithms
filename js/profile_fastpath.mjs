import process from "node:process";
import { performance } from "node:perf_hooks";
import { existsSync, readFileSync, statSync } from "node:fs";

import {
  buildPackedReferenceFromUV,
  dealiasSweepJH01VelocityPacked,
  dealiasSweepRegionGraphVelocityPacked,
  dealiasSweepVariationalVelocityPacked,
  dealiasSweepZW06VelocityPacked,
  setOpenDealiasBackend,
  wrapToNyquist,
} from "./dist/open_dealias.js";
import {
  createOpenDealiasBackendFromModule,
  createOpenDealiasWasmSweepWorkspaceFromModule,
  registerOpenDealiasWasmInitOutput,
} from "./dist/open_dealias_wasm.js";

const mode = process.argv[2] ?? "half";
const shape = mode === "full"
  ? { rows: 720, cols: 1832 }
  : mode === "fragmented"
    ? { rows: 720, cols: 1192 }
    : { rows: 360, cols: 916 };

async function loadProfileModule() {
  const candidates = [
    {
      path: "./pkg-node/open_dealias_wasm.js",
      fullPath: new URL("./pkg-node/open_dealias_wasm.js", import.meta.url),
      wasmUrl: new URL("./pkg-node/open_dealias_wasm_bg.wasm", import.meta.url),
    },
    {
      path: "./pkg/open_dealias_wasm.js",
      fullPath: new URL("./pkg/open_dealias_wasm.js", import.meta.url),
      wasmUrl: new URL("./pkg/open_dealias_wasm_bg.wasm", import.meta.url),
    },
  ].filter((candidate) => existsSync(candidate.fullPath));
  if (candidates.length === 0) {
    throw new Error("no generated wasm package found under js/pkg or js/pkg-node");
  }
  candidates.sort((left, right) => statSync(right.fullPath).mtimeMs - statSync(left.fullPath).mtimeMs);
  return { module: await import(candidates[0].path), wasmUrl: candidates[0].wasmUrl };
}

function cpuMs(start) {
  const diff = process.cpuUsage(start);
  return (diff.user + diff.system) / 1000;
}

function round(value) {
  return Number(value.toFixed(2));
}

async function measureAdapter(name, run) {
  let warmup = run();
  warmup = null;
  global.gc?.();
  const cpuBefore = process.cpuUsage();
  const t0 = performance.now();
  const result = run();
  const t1 = performance.now();
  return {
    name,
    wallMs: round(t1 - t0),
    cpuMs: round(cpuMs(cpuBefore)),
    resultMB: round(result.velocity.byteLength / (1024 * 1024)),
    metadata: result.metadata ?? {},
  };
}

function measureWorkspaceCall(name, workspace, observedF32, runner) {
  runner();
  global.gc?.();

  const t0 = performance.now();
  workspace.observedView.set(observedF32);
  const t1 = performance.now();
  const timings = runner();
  const t2 = performance.now();
  const outputView = workspace.velocityView;
  const sample = outputView.length ? outputView[0] : Number.NaN;
  const t3 = performance.now();

  return {
    name,
    marshalInMs: round((t1 - t0) + timings.marshalInMs),
    wasmMs: round(timings.wasmMs),
    marshalOutMs: round((t3 - t2) + timings.marshalOutMs),
    totalMs: round(t3 - t0),
    outputViewMB: round(outputView.byteLength / (1024 * 1024)),
    sample: round(Number.isFinite(sample) ? sample : 0),
    metadata: workspace.workspaceMetadata?.() ?? {},
  };
}

function addRegionGraphDerivedMetadata(result) {
  const metadata = { ...(result.metadata ?? {}) };
  if (metadata.method === "region_graph_sweep") {
    const seedable = Number(metadata.seedable_region_count ?? 0);
    const pruned = Number(metadata.pruned_disconnected_seedable_regions ?? 0);
    metadata.active_seedable_region_count = Math.max(0, seedable - pruned);
  }
  return { ...result, metadata };
}

const { module: rawModule, wasmUrl } = await loadProfileModule();
if (typeof rawModule.default === "function") {
  const initOutput = await rawModule.default({ module_or_path: readFileSync(wasmUrl) });
  registerOpenDealiasWasmInitOutput(rawModule, initOutput);
}
setOpenDealiasBackend(createOpenDealiasBackendFromModule(rawModule));

const nyquist = 26.12;
const azimuthDeg = Array.from({ length: shape.rows }, (_, index) => (360 * index) / shape.rows);
const referenceField = buildPackedReferenceFromUV(azimuthDeg, shape.cols, 18, -6, 0.5);
let reference = referenceField;
let regionGraphOptions = { reference };
let observed = {
  ...referenceField,
  data: Float64Array.from(referenceField.data, (value, idx) => {
    const shear = ((idx % shape.cols) / shape.cols - 0.5) * 18;
    return wrapToNyquist(value + shear, nyquist);
  }),
};
let previous = {
  ...referenceField,
  data: Float64Array.from(referenceField.data, (value) => value + 1.25),
};

if (mode === "fragmented") {
  reference = undefined;
  regionGraphOptions = { minRegionArea: 4, minValidFraction: 0.15 };
  const data = Float64Array.from({ length: shape.rows * shape.cols }, () => Number.NaN);
  const source = referenceField.data;
  const writeRect = (r0, r1, c0, c1, offset = 0) => {
    for (let row = r0; row < r1; row += 1) {
      for (let col = c0; col < c1; col += 1) {
        const idx = row * shape.cols + col;
        data[idx] = wrapToNyquist(source[idx] + offset, nyquist);
      }
    }
  };
  // Large primary component.
  writeRect(60, 420, 80, 360, 10);
  writeRect(140, 520, 360, 430, 6);
  writeRect(260, 340, 430, 640, 14);
  // Disconnected seedable islands that should now be pruned without a reference.
  writeRect(80, 180, 760, 860, -12);
  writeRect(260, 360, 880, 980, -8);
  writeRect(500, 620, 980, 1100, 16);
  // Sparse fragments that should be skipped entirely.
  for (let block = 0; block < 60; block += 1) {
    const row = (block * 11) % (shape.rows - 8);
    const col = 680 + ((block * 29) % (shape.cols - 688));
    const idxA = row * shape.cols + col;
    const idxB = (row + 1) * shape.cols + col;
    data[idxA] = wrapToNyquist(source[idxA] + 4, nyquist);
    data[idxB] = wrapToNyquist(source[idxB] - 4, nyquist);
  }
  observed = {
    ...referenceField,
    data,
  };
  previous = {
    ...referenceField,
    data: Float64Array.from(data, (value) => (Number.isFinite(value) ? value : Number.NaN)),
  };
}
const observedF32 = Float32Array.from(observed.data, (value) => Math.fround(value));

const adapterResults = [
  addRegionGraphDerivedMetadata(await measureAdapter("region_graph_velocity", () => dealiasSweepRegionGraphVelocityPacked(observed, nyquist, regionGraphOptions))),
  await measureAdapter("zw06_velocity", () => dealiasSweepZW06VelocityPacked(observed, nyquist, reference)),
  await measureAdapter("variational_velocity", () => dealiasSweepVariationalVelocityPacked(observed, nyquist, { reference })),
  await measureAdapter("jh01_velocity", () => dealiasSweepJH01VelocityPacked(observed, nyquist, previous)),
];

const workspace = createOpenDealiasWasmSweepWorkspaceFromModule(rawModule, shape.rows, shape.cols);
let workspaceResults = {
  available: false,
  reason: "module does not expose a persistent sweep workspace API",
  results: [],
};

if (workspace) {
  const results = [];
  if (workspace.supportsVelocityAlgorithm("zw06")) {
    results.push(measureWorkspaceCall(
      "zw06_velocity_workspace",
      workspace,
      observedF32,
      () => workspace.runZw06Velocity(nyquist, reference),
    ));
  }
  if (workspace.supportsVelocityAlgorithm("region_graph")) {
    results.push(measureWorkspaceCall(
      "region_graph_velocity_workspace",
      workspace,
      observedF32,
      () => workspace.runRegionGraphVelocity(nyquist, regionGraphOptions),
    ));
  }
  if (workspace.supportsVelocityAlgorithm("variational")) {
    results.push(measureWorkspaceCall(
      "variational_velocity_workspace",
      workspace,
      observedF32,
      () => workspace.runVariationalVelocity(nyquist, { reference }),
    ));
  }
  if (workspace.supportsVelocityAlgorithm("jh01")) {
    results.push(measureWorkspaceCall(
      "jh01_velocity_workspace",
      workspace,
      observedF32,
      () => workspace.runJH01Velocity(nyquist, previous),
    ));
  }
  workspaceResults = {
    available: true,
    reason: null,
    results,
  };
  workspace.free();
}

console.log(JSON.stringify({
  mode,
  rows: shape.rows,
  cols: shape.cols,
  gates: shape.rows * shape.cols,
  adapterResults,
  workspaceResults,
}, null, 2));
