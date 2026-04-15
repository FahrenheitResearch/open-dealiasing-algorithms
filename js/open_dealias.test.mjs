import test from "node:test";
import assert from "node:assert/strict";

import { buildReferenceFromUV, dealiasRadialES90, dealiasSweepJH01, dealiasSweepZW06, wrapToNyquist } from "./dist/open_dealias.js";

test("wrapToNyquist keeps values inside interval", () => {
  assert.equal(wrapToNyquist(27, 10), 7);
  assert.equal(wrapToNyquist(-27, 10), -7);
});

test("dealiasRadialES90 restores a smooth folded radial", () => {
  const truth = Array.from({ length: 60 }, (_, i) => -18 + i * 0.8);
  const observed = truth.map((v) => wrapToNyquist(v, 10));
  const res = dealiasRadialES90(observed, 10);
  const mae = res.velocity.reduce((acc, v, i) => acc + Math.abs(v - truth[i]), 0) / truth.length;
  assert.ok(mae < 1.5);
});

test("dealiasSweepZW06 solves a broad folded background flow", () => {
  const azimuth = Array.from({ length: 90 }, (_, i) => i * 4);
  const ref = buildReferenceFromUV(azimuth, 40, 18, 0);
  const observed = ref.map((row) => row.map((v) => wrapToNyquist(v, 10)));
  const res = dealiasSweepZW06(observed, 10, ref);
  let mae = 0;
  let n = 0;
  for (let i = 0; i < observed.length; i++) {
    for (let j = 0; j < observed[0].length; j++) {
      mae += Math.abs(res.velocity[i][j] - ref[i][j]);
      n += 1;
    }
  }
  mae /= n;
  assert.ok(mae < 0.2);
});

test("dealiasSweepJH01 uses previous volume as anchor", () => {
  const azimuth = Array.from({ length: 90 }, (_, i) => i * 4);
  const prev = buildReferenceFromUV(azimuth, 30, 16, 2);
  const currentTruth = prev.map((row) => row.slice());
  const currentObserved = currentTruth.map((row) => row.map((v) => wrapToNyquist(v, 10)));
  const res = dealiasSweepJH01(currentObserved, 10, prev);
  let mae = 0;
  let n = 0;
  for (let i = 0; i < currentTruth.length; i++) {
    for (let j = 0; j < currentTruth[0].length; j++) {
      mae += Math.abs(res.velocity[i][j] - currentTruth[i][j]);
      n += 1;
    }
  }
  mae /= n;
  assert.ok(mae < 0.2);
});
