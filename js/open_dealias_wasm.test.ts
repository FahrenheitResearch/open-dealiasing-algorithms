import test from "node:test";
import assert from "node:assert/strict";

import { buildPackedReferenceFromUV, wrapToNyquist, type PackedSweep } from "./open_dealias.js";
import { createOpenDealiasBackendFromModule } from "./open_dealias_wasm.js";

test("createOpenDealiasBackendFromModule adapts sweep results", () => {
  const backend = createOpenDealiasBackendFromModule({
    dealiasSweepZw06(
      observed: Float64Array,
      rows: number,
      cols: number,
      _nyquist: number,
      _reference: Float64Array,
    ) {
      return {
        velocity: Float64Array.from(observed, (value) => value + 1),
        folds: Int16Array.from({ length: observed.length }, () => 0),
        confidence: Float32Array.from({ length: observed.length }, () => 0.75),
        reference: new Float64Array(observed.length),
        rows,
        cols,
        metadata_json: JSON.stringify({ method: "zw06", backend: "wasm-test" }),
      };
    },
  });

  const observed: PackedSweep = {
    data: new Float64Array([1, 2, 3, 4]),
    azimuthCount: 2,
    gateCount: 2,
  };
  const result = backend.dealiasSweepZW06Packed!(observed, 10);
  assert.deepEqual(Array.from(result.velocity), [2, 3, 4, 5]);
  assert.equal(result.azimuthCount, 2);
  assert.equal(result.gateCount, 2);
  assert.equal(result.metadata.backend, "wasm-test");
});

test("createOpenDealiasBackendFromModule adapts xu11 reference packing", () => {
  const reference = buildPackedReferenceFromUV([0, 180], 2, 10, 0);
  const backend = createOpenDealiasBackendFromModule({
    dealiasSweepXu11(
      observed: Float64Array,
      rows: number,
      cols: number,
      _nyquist: number,
      _azimuthDeg: number[],
      _elevationDeg: number,
      externalReference: Float64Array,
    ) {
      assert.equal(externalReference.length, observed.length);
      return {
        velocity: new Float64Array(observed),
        folds: new Int16Array(observed.length),
        confidence: Float32Array.from({ length: observed.length }, () => 1),
        reference: new Float64Array(externalReference),
        rows,
        cols,
        metadata_json: JSON.stringify({ method: "xu11" }),
      };
    },
  });

  const observed: PackedSweep = {
    data: Float64Array.from(reference.data, (value) => wrapToNyquist(value, 8)),
    azimuthCount: reference.azimuthCount,
    gateCount: reference.gateCount,
  };
  const result = backend.dealiasSweepXu11Packed!(observed, 8, {
    azimuthDeg: [0, 180],
    reference,
  });
  assert.equal(result.metadata.method, "xu11");
  assert.equal(result.velocity.length, observed.data.length);
});
