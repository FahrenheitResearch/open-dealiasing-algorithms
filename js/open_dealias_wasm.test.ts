import test from "node:test";
import assert from "node:assert/strict";

import { buildPackedReferenceFromUV, wrapToNyquist, type PackedSweep } from "./open_dealias.js";
import {
  createOpenDealiasBackendFromModule,
  createOpenDealiasWasmSweepWorkspaceFromModule,
  supportsOpenDealiasWasmSweepWorkspace,
} from "./open_dealias_wasm.js";

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

test("createOpenDealiasBackendFromModule adapts velocity-only sweep results", () => {
  const backend = createOpenDealiasBackendFromModule({
    dealiasSweepRegionGraphVelocity(
      observed: Float64Array,
      rows: number,
      cols: number,
    ) {
      return {
        velocity: Float32Array.from(observed, (value) => value + 2),
        rows,
        cols,
        metadata_json: JSON.stringify({ method: "region_graph", output: "velocity_only" }),
      };
    },
  });

  const observed: PackedSweep = {
    data: new Float64Array([1, 2, 3, 4]),
    azimuthCount: 2,
    gateCount: 2,
  };
  const result = backend.dealiasSweepRegionGraphVelocityPacked!(observed, 10);
  assert.deepEqual(Array.from(result.velocity), [3, 4, 5, 6]);
  assert.equal(result.velocity.constructor.name, "Float32Array");
  assert.equal(result.metadata.output, "velocity_only");
});

test("workspace bridge exposes direct views when persistent sweep buffers are available", () => {
  const memory = new WebAssembly.Memory({ initial: 1 });
  class SweepVelocityWorkspace {
    rows: number;
    cols: number;
    len: number;
    constructor(rows: number, cols: number) {
      this.rows = rows;
      this.cols = cols;
      this.len = rows * cols;
    }
    observed_ptr() {
      return 0;
    }
    reference_ptr() {
      return this.len * 4;
    }
    velocity_ptr() {
      return this.len * 8;
    }
    metadata_json() {
      return JSON.stringify({ workspace: true });
    }
    run_zw06_velocity_only(_nyquist: number) {
      const observed = new Float32Array(memory.buffer, 0, this.len);
      const reference = new Float32Array(memory.buffer, this.len * 4, this.len);
      const velocity = new Float32Array(memory.buffer, this.len * 8, this.len);
      for (let index = 0; index < this.len; index++) {
        velocity[index] = observed[index] + (Number.isFinite(reference[index]) ? reference[index] : 0) + 1;
      }
    }
    free() {}
  }

  const fakeModule = {
    memory,
    SweepVelocityWorkspace,
    dealiasSweepZw06Velocity() {
      throw new Error("expected workspace path");
    },
  };
  assert.equal(supportsOpenDealiasWasmSweepWorkspace(fakeModule), true);

  const workspace = createOpenDealiasWasmSweepWorkspaceFromModule(fakeModule, 2, 2);
  assert.ok(workspace);
  if (!workspace) {
    throw new Error("expected workspace bridge");
  }
  assert.equal(workspace.supportsVelocityAlgorithm("zw06"), true);
  assert.equal(workspace.supportsVelocityAlgorithm("region_graph"), false);

  workspace.observedView.set([1, 2, 3, 4]);
  const reference = {
    data: new Float64Array([10, 20, 30, 40]),
    azimuthCount: 2,
    gateCount: 2,
  };
  const timings = workspace.runZw06Velocity(10, reference);
  assert.ok(timings.totalMs >= 0);
  assert.deepEqual(Array.from(workspace.velocityView), [12, 23, 34, 45]);
  assert.deepEqual(Array.from(workspace.snapshotVelocity()), [12, 23, 34, 45]);
  workspace.free();

  const backend = createOpenDealiasBackendFromModule(fakeModule);
  const observed: PackedSweep = {
    data: new Float64Array([1, 2, 3, 4]),
    azimuthCount: 2,
    gateCount: 2,
  };
  const backendResult = backend.dealiasSweepZW06VelocityPacked!(observed, 10, reference);
  assert.deepEqual(Array.from(backendResult.velocity), [12, 23, 34, 45]);
  assert.equal(backendResult.metadata.path, "workspace");
  assert.equal(backendResult.metadata.workspace, true);
});
