import test from "node:test";
import assert from "node:assert/strict";

import {
  buildPackedReferenceFromUV,
  buildReferenceFromUV,
  dealiasDualPrfPacked,
  dealiasRadialES90,
  dealiasSweepJH01,
  dealiasSweepJH01VelocityPacked,
  dealiasSweepRegionGraphPacked,
  dealiasSweepRegionGraphVelocityPacked,
  dealiasSweepVariationalPacked,
  dealiasSweepVariationalVelocityPacked,
  dealiasSweepZW06,
  dealiasSweepZW06Packed,
  initOpenDealiasWasm,
  packSweep,
  resetOpenDealiasBackend,
  setOpenDealiasBackend,
  sanitizePackedSweep,
  unpackSweep,
  wrapToNyquist,
} from "./open_dealias.js";
import { createOpenDealiasBackendFromModule } from "./open_dealias_wasm.js";

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

test("packSweep and unpackSweep round-trip typed arrays", () => {
  const matrix = [
    [1, 2, 3],
    [4, 5, 6],
  ];
  const packed = packSweep(matrix);
  assert.equal(packed.data.constructor.name, "Float64Array");
  assert.deepEqual(unpackSweep(packed), matrix);
});

test("packSweep reuses already-packed typed arrays", () => {
  const packed = {
    data: new Float64Array([1, 2, 3, 4]),
    azimuthCount: 2,
    gateCount: 2,
  };
  const repacked = packSweep(packed);
  assert.equal(repacked.data, packed.data);
});

test("packSweep can normalize finite no-data sentinels to NaN", () => {
  const packed = packSweep(
    [
      [0, -9999, 2],
      [3, 4, 0],
    ],
    { zeroIsMissing: true, noDataValues: [-9999] },
  );
  assert.ok(Number.isNaN(packed.data[0]));
  assert.ok(Number.isNaN(packed.data[1]));
  assert.equal(packed.data[2], 2);
  assert.ok(Number.isNaN(packed.data[5]));
});

test("sanitizePackedSweep normalizes already-packed sentinel values", () => {
  const observed = {
    data: new Float64Array([0, -9999, 2, 3]),
    azimuthCount: 2,
    gateCount: 2,
  };
  const sanitized = sanitizePackedSweep(observed, { zeroIsMissing: true, noDataValue: -9999 });
  assert.notEqual(sanitized.data, observed.data);
  assert.ok(Number.isNaN(sanitized.data[0]));
  assert.ok(Number.isNaN(sanitized.data[1]));
  assert.equal(sanitized.data[2], 2);
});

test("dealiasSweepZW06Packed returns flat typed arrays", () => {
  const azimuth = Array.from({ length: 12 }, (_, i) => i * 30);
  const ref = buildPackedReferenceFromUV(azimuth, 8, 15, -2);
  const observed = {
    ...ref,
    data: Float64Array.from(ref.data, (value) => wrapToNyquist(value, 10)),
  };
  const result = dealiasSweepZW06Packed(observed, 10, ref);
  assert.equal(result.velocity.constructor.name, "Float64Array");
  assert.equal(result.folds.constructor.name, "Int16Array");
  assert.equal(result.confidence.constructor.name, "Float32Array");
  assert.equal(result.azimuthCount, 12);
  assert.equal(result.gateCount, 8);
});

test("velocity-only packed helpers return compact typed arrays", () => {
  const azimuth = Array.from({ length: 12 }, (_, i) => i * 30);
  const ref = buildPackedReferenceFromUV(azimuth, 8, 15, -2);
  const observed = {
    ...ref,
    data: Float64Array.from(ref.data, (value) => wrapToNyquist(value, 10)),
  };
  const region = dealiasSweepRegionGraphVelocityPacked(observed, 10, { reference: ref });
  const jh01 = dealiasSweepJH01VelocityPacked(observed, 10, ref);
  assert.equal(region.velocity.constructor.name, "Float32Array");
  assert.equal(region.velocity.length, observed.data.length);
  assert.equal(jh01.velocity.constructor.name, "Float32Array");
  assert.equal(jh01.velocity.length, observed.data.length);
});

test("initOpenDealiasWasm accepts a module-style backend factory", async () => {
  await initOpenDealiasWasm(async () => ({
    async default() {
      return undefined;
    },
    createOpenDealiasBackend() {
      return {
        name: "fake-wasm",
        kind: "wasm" as const,
        dealiasSweepRegionGraphPacked(observed: { data: Float64Array }) {
          return {
            velocity: Float64Array.from(observed.data, (value) => value + 5),
            folds: new Int16Array(observed.data.length),
            confidence: Float32Array.from({ length: observed.data.length }, () => 1),
            metadata: { method: "region_graph", backend: "fake-wasm" },
          };
        },
      };
    },
  }));

  const observed = packSweep([[1, 2], [3, 4]]);
  const result = dealiasSweepRegionGraphPacked(observed, 5);
  assert.equal(result.metadata.backend, "fake-wasm");
  assert.deepEqual(Array.from(result.velocity), [6, 7, 8, 9]);

  resetOpenDealiasBackend();
});

test("WASM variational backend defaults to safer ZW06 bootstrap", () => {
  const calls: string[] = [];
  const fakeModule = {
    dealiasSweepZw06Velocity(_observed: Float64Array, rows: number, cols: number) {
      calls.push("zw06");
      return {
        velocity: Float32Array.from({ length: rows * cols }, (_, index) => index + 10),
        rows,
        cols,
        metadata_json: JSON.stringify({ method: "zw06" }),
        free() {},
      };
    },
    dealiasSweepVariationalRefine(_observed: Float64Array, rows: number, cols: number, initialCorrected: Float64Array) {
      calls.push("refine");
      assert.deepEqual(Array.from(initialCorrected), [10, 11, 12, 13]);
      return {
        velocity: Float64Array.from(initialCorrected, (value) => value + 1),
        folds: new Int16Array(rows * cols),
        confidence: Float32Array.from({ length: rows * cols }, () => 1),
        rows,
        cols,
        metadata_json: JSON.stringify({ method: "variational_refine" }),
        free() {},
      };
    },
    dealiasSweepVariational() {
      calls.push("direct");
      return {
        velocity: new Float64Array([99, 99, 99, 99]),
        folds: new Int16Array(4),
        confidence: Float32Array.from({ length: 4 }, () => 1),
        rows: 2,
        cols: 2,
        metadata_json: JSON.stringify({ method: "variational_direct" }),
        free() {},
      };
    },
  };

  setOpenDealiasBackend(createOpenDealiasBackendFromModule(fakeModule));
  try {
    const result = dealiasSweepVariationalPacked([[1, 2], [3, 4]], 10);
    assert.deepEqual(calls, ["zw06", "refine"]);
    assert.deepEqual(Array.from(result.velocity), [11, 12, 13, 14]);
    assert.equal(result.metadata.method, "variational_refine");
    assert.equal(result.metadata.bootstrap_method, "zw06");
  } finally {
    resetOpenDealiasBackend();
  }
});

test("WASM variational backend still supports explicit region-graph bootstrap", () => {
  const calls: string[] = [];
  const fakeModule = {
    dealiasSweepVariationalVelocity(_observed: Float64Array, rows: number, cols: number) {
      calls.push("direct");
      return {
        velocity: new Float32Array([21, 22, 23, 24]),
        rows,
        cols,
        metadata_json: JSON.stringify({ method: "variational_direct" }),
        free() {},
      };
    },
    dealiasSweepZw06Velocity() {
      calls.push("zw06");
      throw new Error("unexpected zw06 bootstrap");
    },
    dealiasSweepVariationalRefine() {
      calls.push("refine");
      throw new Error("unexpected refine path");
    },
  };

  setOpenDealiasBackend(createOpenDealiasBackendFromModule(fakeModule));
  try {
    const result = dealiasSweepVariationalVelocityPacked([[1, 2], [3, 4]], 10, {
      bootstrapMethod: "region_graph",
    });
    assert.deepEqual(calls, ["direct"]);
    assert.deepEqual(Array.from(result.velocity), [21, 22, 23, 24]);
    assert.equal(result.metadata.method, "variational_direct");
    assert.equal(result.metadata.bootstrap_method, "region_graph");
  } finally {
    resetOpenDealiasBackend();
  }
});

test("dealiasDualPrfPacked resolves a folded paired field", () => {
  const truth = [
    [18, 21],
    [24, 27],
  ];
  const low = truth.map((row) => row.map((value) => wrapToNyquist(value, 10)));
  const high = truth.map((row) => row.map((value) => wrapToNyquist(value, 14)));
  const result = dealiasDualPrfPacked(low, high, 10, 14);
  assert.equal(result.velocity.length, 4);
  assert.ok(Array.from(result.confidence).every((value) => value >= 0));
});
