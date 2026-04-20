import {
  buildPackedReferenceFromUV,
  normalizeRegionGraphOptions,
  normalizeVariationalOptions,
  packSweep,
  packVolume,
  type DualPrfOptions,
  type JH01Options,
  type MlOptions,
  type OpenDealiasBackend,
  type PackedDealiasResult,
  type PackedSweep,
  type PackedVelocityResult,
  type PackedVolume,
  type PackedVolumeDealiasResult,
  type RecursiveOptions,
  type RegionGraphOptions,
  type VariationalOptions,
  type VolumeOptions,
  type Xu11Options,
} from "./open_dealias.js";

type RawWasmModule = Record<string, unknown>;

let rawModulePromise: Promise<RawWasmModule> | null = null;
const dynamicImport = new Function("path", "return import(path);") as (path: string) => Promise<RawWasmModule>;

function parseMetadataJson(value: unknown): Record<string, unknown> {
  if (typeof value !== "string" || value.length === 0) {
    return {};
  }
  try {
    const parsed = JSON.parse(value);
    return typeof parsed === "object" && parsed !== null ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
}

function maybeFree(raw: Record<string, unknown>): void {
  const free = raw.free;
  if (typeof free === "function") {
    try {
      free.call(raw);
    } catch {
      // Ignore finalizer/free errors from non-owning shims.
    }
  }
}

function normalizeSweepResult(raw: Record<string, unknown>): PackedDealiasResult {
  const result = {
    velocity: raw.velocity instanceof Float64Array ? raw.velocity : new Float64Array(raw.velocity as ArrayLike<number>),
    folds: raw.folds instanceof Int16Array ? raw.folds : Int16Array.from(raw.folds as ArrayLike<number>),
    confidence: raw.confidence instanceof Float32Array ? raw.confidence : Float32Array.from(raw.confidence as ArrayLike<number>),
    azimuthCount: Number(raw.rows),
    gateCount: Number(raw.cols),
    metadata: parseMetadataJson(raw.metadata_json),
  };
  maybeFree(raw);
  return result;
}

function normalizeVolumeResult(raw: Record<string, unknown>): PackedVolumeDealiasResult {
  const result = {
    velocity: raw.velocity instanceof Float64Array ? raw.velocity : new Float64Array(raw.velocity as ArrayLike<number>),
    folds: raw.folds instanceof Int16Array ? raw.folds : Int16Array.from(raw.folds as ArrayLike<number>),
    confidence: raw.confidence instanceof Float32Array ? raw.confidence : Float32Array.from(raw.confidence as ArrayLike<number>),
    sweepCount: Number(raw.sweeps),
    azimuthCount: Number(raw.rows),
    gateCount: Number(raw.cols),
    metadata: parseMetadataJson(raw.metadata_json),
  };
  maybeFree(raw);
  return result;
}

function normalizeVelocityResult(raw: Record<string, unknown>): PackedVelocityResult {
  const result = {
    velocity: raw.velocity instanceof Float32Array ? raw.velocity : Float32Array.from(raw.velocity as ArrayLike<number>),
    azimuthCount: Number(raw.rows),
    gateCount: Number(raw.cols),
    metadata: parseMetadataJson(raw.metadata_json),
  };
  maybeFree(raw);
  return result;
}

function normalizeSweepVelocityFromRefine(raw: Record<string, unknown>): PackedVelocityResult {
  const result = normalizeSweepResult(raw);
  return {
    velocity: Float32Array.from(result.velocity),
    azimuthCount: result.azimuthCount,
    gateCount: result.gateCount,
    metadata: { ...result.metadata, output: "velocity_only" },
  };
}

function normalizeNyquist(nyquist: number | ArrayLike<number>, sweepCount: number): number[] {
  if (typeof nyquist === "number") {
    return [nyquist];
  }
  return Array.from({ length: sweepCount }, (_, index) => {
    const value = nyquist[index];
    if (typeof value !== "number") {
      throw new Error(`missing nyquist for sweep ${index}`);
    }
    return value;
  });
}

function requireFunction<T extends (...args: any[]) => any>(module: RawWasmModule, name: string): T {
  const candidate = module[name];
  if (typeof candidate !== "function") {
    throw new Error(`wasm module is missing ${name}`);
  }
  return candidate as T;
}

async function loadRawModule(): Promise<RawWasmModule> {
  if (rawModulePromise === null) {
    rawModulePromise = dynamicImport("../pkg/open_dealias_wasm.js");
  }
  return rawModulePromise;
}

export async function loadRawOpenDealiasWasmModule(): Promise<RawWasmModule> {
  return loadRawModule();
}

export default async function init(input?: unknown): Promise<unknown> {
  const module = await loadRawModule();
  const initFn = requireFunction<(arg?: unknown) => Promise<unknown> | unknown>(module, "default");
  return initFn(input);
}

export async function createOpenDealiasBackend(): Promise<OpenDealiasBackend> {
  const module = await loadRawModule();
  await init();
  return createOpenDealiasBackendFromModule(module);
}

export function createOpenDealiasBackendFromModule(module: RawWasmModule): OpenDealiasBackend {
  const call = <T>(name: string, ...args: unknown[]): T =>
    requireFunction<(...callArgs: unknown[]) => T>(module, name)(...args);

  const callRegionGraph = (observed: PackedSweep, nyquist: number, options?: RegionGraphOptions): PackedDealiasResult => {
    const normalized = normalizeRegionGraphOptions(options);
    const reference = normalized?.reference ? packSweep(normalized.reference) : undefined;
    return normalizeSweepResult(call<Record<string, unknown>>(
      "dealiasSweepRegionGraph",
      observed.data,
      observed.azimuthCount,
      observed.gateCount,
      nyquist,
      reference?.data ?? [],
      normalized?.blockRows,
      normalized?.blockCols,
      normalized?.referenceWeight ?? 0.75,
      normalized?.maxIterations ?? 6,
      normalized?.maxAbsFold ?? 8,
      normalized?.wrapAzimuth ?? true,
      normalized?.minRegionArea ?? 4,
      normalized?.minValidFraction ?? 0.15,
    ));
  };

  const callRegionGraphVelocity = (observed: PackedSweep, nyquist: number, options?: RegionGraphOptions): PackedVelocityResult => {
    const normalized = normalizeRegionGraphOptions(options);
    const reference = normalized?.reference ? packSweep(normalized.reference) : undefined;
    return normalizeVelocityResult(call<Record<string, unknown>>(
      "dealiasSweepRegionGraphVelocity",
      observed.data,
      observed.azimuthCount,
      observed.gateCount,
      nyquist,
      reference?.data ?? [],
      normalized?.blockRows,
      normalized?.blockCols,
      normalized?.referenceWeight ?? 0.75,
      normalized?.maxIterations ?? 6,
      normalized?.maxAbsFold ?? 8,
      normalized?.wrapAzimuth ?? true,
      normalized?.minRegionArea ?? 4,
      normalized?.minValidFraction ?? 0.15,
    ));
  };

  const callVariational = (observed: PackedSweep, nyquist: number, options?: VariationalOptions): PackedDealiasResult => {
    const normalized = normalizeVariationalOptions(options);
    const reference = normalized.reference ? packSweep(normalized.reference) : undefined;
    if (normalized.bootstrapMethod === "region_graph") {
      const result = normalizeSweepResult(call<Record<string, unknown>>(
        "dealiasSweepVariational",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        reference?.data ?? [],
        normalized.blockRows,
        normalized.blockCols,
        normalized.bootstrapReferenceWeight ?? 0.75,
        normalized.bootstrapIterations ?? 6,
        normalized.bootstrapMaxAbsFold ?? 8,
        normalized.minRegionArea ?? 4,
        normalized.minValidFraction ?? 0.15,
        normalized.maxAbsFold ?? 8,
        normalized.neighborWeight ?? 1.0,
        normalized.referenceWeight ?? 0.50,
        normalized.smoothnessWeight ?? 0.20,
        normalized.maxIterations ?? 8,
        normalized.wrapAzimuth ?? true,
      ));
      return {
        ...result,
        metadata: { ...result.metadata, bootstrap_method: "region_graph" },
      };
    }

    const bootstrap = call<Record<string, unknown>>(
      "dealiasSweepZw06Velocity",
      observed.data,
      observed.azimuthCount,
      observed.gateCount,
      nyquist,
      reference?.data ?? [],
      normalized.bootstrapWeakThresholdFraction ?? 0.35,
      normalized.wrapAzimuth ?? true,
      normalized.bootstrapMaxIterationsPerPass ?? 12,
      normalized.bootstrapIncludeDiagonals ?? true,
      normalized.bootstrapRecenterWithoutReference ?? true,
    );
    const initialCorrected = bootstrap.velocity instanceof Float32Array
      ? Float64Array.from(bootstrap.velocity)
      : Float64Array.from(bootstrap.velocity as ArrayLike<number>);
    maybeFree(bootstrap);
    const result = normalizeSweepResult(call<Record<string, unknown>>(
      "dealiasSweepVariationalRefine",
      observed.data,
      observed.azimuthCount,
      observed.gateCount,
      initialCorrected,
      nyquist,
      reference?.data ?? [],
      normalized.maxAbsFold ?? 8,
      normalized.neighborWeight ?? 1.0,
      normalized.referenceWeight ?? 0.50,
      normalized.smoothnessWeight ?? 0.20,
      normalized.maxIterations ?? 8,
      normalized.wrapAzimuth ?? true,
    ));
    return {
      ...result,
      metadata: { ...result.metadata, bootstrap_method: "zw06" },
    };
  };

  const callVariationalVelocity = (observed: PackedSweep, nyquist: number, options?: VariationalOptions): PackedVelocityResult => {
    const normalized = normalizeVariationalOptions(options);
    const reference = normalized.reference ? packSweep(normalized.reference) : undefined;
    if (normalized.bootstrapMethod === "region_graph") {
      const result = normalizeVelocityResult(call<Record<string, unknown>>(
        "dealiasSweepVariationalVelocity",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        reference?.data ?? [],
        normalized.blockRows,
        normalized.blockCols,
        normalized.minRegionArea ?? 4,
        normalized.minValidFraction ?? 0.15,
        normalized.maxAbsFold ?? 8,
        normalized.neighborWeight ?? 1.0,
        normalized.referenceWeight ?? 0.50,
        normalized.smoothnessWeight ?? 0.20,
        normalized.maxIterations ?? 8,
        normalized.wrapAzimuth ?? true,
      ));
      return {
        ...result,
        metadata: { ...result.metadata, bootstrap_method: "region_graph" },
      };
    }

    const bootstrap = call<Record<string, unknown>>(
      "dealiasSweepZw06Velocity",
      observed.data,
      observed.azimuthCount,
      observed.gateCount,
      nyquist,
      reference?.data ?? [],
      normalized.bootstrapWeakThresholdFraction ?? 0.35,
      normalized.wrapAzimuth ?? true,
      normalized.bootstrapMaxIterationsPerPass ?? 12,
      normalized.bootstrapIncludeDiagonals ?? true,
      normalized.bootstrapRecenterWithoutReference ?? true,
    );
    const initialCorrected = bootstrap.velocity instanceof Float32Array
      ? Float64Array.from(bootstrap.velocity)
      : Float64Array.from(bootstrap.velocity as ArrayLike<number>);
    maybeFree(bootstrap);
    const result = normalizeSweepVelocityFromRefine(call<Record<string, unknown>>(
      "dealiasSweepVariationalRefine",
      observed.data,
      observed.azimuthCount,
      observed.gateCount,
      initialCorrected,
      nyquist,
      reference?.data ?? [],
      normalized.maxAbsFold ?? 8,
      normalized.neighborWeight ?? 1.0,
      normalized.referenceWeight ?? 0.50,
      normalized.smoothnessWeight ?? 0.20,
      normalized.maxIterations ?? 8,
      normalized.wrapAzimuth ?? true,
    ));
    return {
      ...result,
      metadata: { ...result.metadata, bootstrap_method: "zw06" },
    };
  };

  return {
    name: "open-dealias-wasm",
    kind: "wasm",
    wrapToNyquist(value, nyquist) {
      return Number(call<ArrayLike<number>>("wrapToNyquistFlat", [value], nyquist)[0]);
    },
    unfoldToReference(observed, reference, nyquist) {
      return Number(call<ArrayLike<number>>("unfoldToReferenceFlat", [observed], [reference], nyquist, 32)[0]);
    },
    foldCount(unfolded, observed, nyquist) {
      return Number(call<ArrayLike<number>>("foldCountsFlat", [unfolded], [observed], nyquist)[0] ?? 0);
    },
    estimateUniformWindVAD(observed, azimuthDeg, elevationDeg = 0) {
      const result = call<{ u: number; v: number }>(
        "estimateUniformWindVad",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        1.0,
        azimuthDeg,
        elevationDeg,
        1.0,
        6,
        0.85,
      );
      return [result.u, result.v] as const;
    },
    dealiasSweepZW06Packed(observed, nyquist, reference) {
      return normalizeSweepResult(call<Record<string, unknown>>(
        "dealiasSweepZw06",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        reference?.data ?? [],
        0.35,
        true,
        12,
        true,
        true,
      ));
    },
    dealiasSweepZW06VelocityPacked(observed, nyquist, reference) {
      return normalizeVelocityResult(call<Record<string, unknown>>(
        "dealiasSweepZw06Velocity",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        reference?.data ?? [],
        0.35,
        true,
        12,
        true,
        true,
      ));
    },
    dealiasSweepXu11Packed(observed, nyquist, options) {
      const reference = options.reference
        ? packSweep(options.reference)
        : options.backgroundUV
          ? buildPackedReferenceFromUV(
              options.azimuthDeg,
              observed.gateCount,
              options.backgroundUV[0],
              options.backgroundUV[1],
              options.elevationDeg ?? 0,
            )
          : undefined;
      return normalizeSweepResult(call<Record<string, unknown>>(
        "dealiasSweepXu11",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        options.azimuthDeg,
        options.elevationDeg ?? 0,
        reference?.data ?? [],
        1.0,
        true,
      ));
    },
    dealiasSweepXu11VelocityPacked(observed, nyquist, options) {
      const reference = options.reference
        ? packSweep(options.reference)
        : options.backgroundUV
          ? buildPackedReferenceFromUV(
              options.azimuthDeg,
              observed.gateCount,
              options.backgroundUV[0],
              options.backgroundUV[1],
              options.elevationDeg ?? 0,
            )
          : undefined;
      return normalizeVelocityResult(call<Record<string, unknown>>(
        "dealiasSweepXu11Velocity",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        options.azimuthDeg,
        options.elevationDeg ?? 0,
        reference?.data ?? [],
        1.0,
        true,
      ));
    },
    dealiasSweepJH01Packed(observed, nyquist, previousCorrected, options) {
      return normalizeSweepResult(call<Record<string, unknown>>(
        "dealiasSweepJH01Packed",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        previousCorrected.data,
        [],
        options?.shiftAz ?? 0,
        options?.shiftRange ?? 0,
        true,
        true,
      ));
    },
    dealiasSweepJH01VelocityPacked(observed, nyquist, previousCorrected, options) {
      return normalizeVelocityResult(call<Record<string, unknown>>(
        "dealiasSweepJH01Velocity",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        previousCorrected.data,
        [],
        options?.shiftAz ?? 0,
        options?.shiftRange ?? 0,
        true,
        true,
      ));
    },
    dealiasSweepRegionGraphPacked(observed, nyquist, options) {
      return callRegionGraph(observed, nyquist, options);
    },
    dealiasSweepRegionGraphVelocityPacked(observed, nyquist, options) {
      return callRegionGraphVelocity(observed, nyquist, options);
    },
    dealiasSweepRecursivePacked(observed, nyquist, options) {
      const reference = options?.reference ? packSweep(options.reference) : undefined;
      return normalizeSweepResult(call<Record<string, unknown>>(
        "dealiasSweepRecursive",
        observed.data,
        observed.azimuthCount,
        observed.gateCount,
        nyquist,
        reference?.data ?? [],
        5,
        24,
        0.60,
        0.70,
        8,
        true,
      ));
    },
    dealiasSweepVariationalPacked(observed, nyquist, options) {
      return callVariational(observed, nyquist, options);
    },
    dealiasSweepVariationalVelocityPacked(observed, nyquist, options) {
      return callVariationalVelocity(observed, nyquist, options);
    },
    dealiasDualPrfPacked(lowObserved, highObserved, lowNyquist, highNyquist, options) {
      return normalizeSweepResult(call<Record<string, unknown>>(
        "dealiasDualPrfPacked",
        lowObserved.data,
        highObserved.data,
        lowObserved.azimuthCount,
        lowObserved.gateCount,
        lowNyquist,
        highNyquist,
        options?.reference ? packSweep(options.reference).data : [],
        options?.maxAbsFold ?? 8,
      ));
    },
    dealiasVolumeJH01Packed(observed, nyquist, previousCorrected, options) {
      const normalizedNyquist = normalizeNyquist(nyquist, observed.sweepCount);
      const previousVolume = options?.previousVolume ? packVolume(options.previousVolume) : previousCorrected;
      const elevationDeg = Array.from({ length: observed.sweepCount }, (_, index) => index as number);
      const azimuthDeg = Array.from({ length: observed.azimuthCount }, (_, index) => index * (360 / Math.max(observed.azimuthCount, 1)));
      return normalizeVolumeResult(call<Record<string, unknown>>(
        "dealiasVolumeJH01Packed",
        observed.data,
        observed.sweepCount,
        observed.azimuthCount,
        observed.gateCount,
        normalizedNyquist,
        azimuthDeg,
        elevationDeg,
        previousVolume.data,
        [],
        [],
        0,
        0,
        true,
      ));
    },
    dealiasVolume3DPacked(observed, nyquist, options) {
      return normalizeVolumeResult(call<Record<string, unknown>>(
        "dealiasVolume3DPacked",
        observed.data,
        observed.sweepCount,
        observed.azimuthCount,
        observed.gateCount,
        normalizeNyquist(nyquist, observed.sweepCount),
        options?.referenceVolume ? packVolume(options.referenceVolume).data : [],
        true,
        4,
      ));
    },
  };
}
