import process from "node:process";
import { performance } from "node:perf_hooks";

import {
  setOpenDealiasBackend,
  buildPackedReferenceFromUV,
  dealiasSweepJH01VelocityPacked,
  dealiasSweepRegionGraphVelocityPacked,
  dealiasSweepZW06VelocityPacked,
  wrapToNyquist,
} from "./dist/open_dealias.js";
import { createOpenDealiasBackendFromModule } from "./dist/open_dealias_wasm.js";

const mode = process.argv[2] ?? "half";
const shape = mode === "full"
  ? { rows: 720, cols: 1832 }
  : { rows: 360, cols: 916 };

const rawModule = await import("./pkg-node/open_dealias_wasm.js");
setOpenDealiasBackend(createOpenDealiasBackendFromModule(rawModule));

const nyquist = 26.12;
const azimuthDeg = Array.from({ length: shape.rows }, (_, i) => (360 * i) / shape.rows);
const reference = buildPackedReferenceFromUV(azimuthDeg, shape.cols, 18, -6, 0.5);
const observed = {
  ...reference,
  data: Float64Array.from(reference.data, (value, idx) => {
    const shear = ((idx % shape.cols) / shape.cols - 0.5) * 18;
    return wrapToNyquist(value + shear, nyquist);
  }),
};
const previous = {
  ...reference,
  data: Float64Array.from(reference.data, (value) => value + 1.25),
};

function cpuMs(start) {
  const diff = process.cpuUsage(start);
  return (diff.user + diff.system) / 1000;
}

async function measure(name, run) {
  let warmup = await run();
  warmup = null;
  global.gc?.();
  const cpuBefore = process.cpuUsage();
  const t0 = performance.now();
  const result = await run();
  const t1 = performance.now();
  return {
    name,
    wallMs: Number((t1 - t0).toFixed(2)),
    cpuMs: Number(cpuMs(cpuBefore).toFixed(2)),
    resultMB: Number((result.velocity.byteLength / (1024 * 1024)).toFixed(2)),
  };
}

const results = [
  await measure("region_graph_velocity", () => dealiasSweepRegionGraphVelocityPacked(observed, nyquist, { reference })),
  await measure("zw06_velocity", () => dealiasSweepZW06VelocityPacked(observed, nyquist, reference)),
  await measure("jh01_velocity", () => dealiasSweepJH01VelocityPacked(observed, nyquist, previous)),
];

console.log(JSON.stringify({
  mode,
  rows: shape.rows,
  cols: shape.cols,
  gates: shape.rows * shape.cols,
  results,
}, null, 2));
