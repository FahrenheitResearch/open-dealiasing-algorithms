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
type RawWasmWorkspace = Record<string, unknown>;
type RawWasmWorkspaceCtor = new (rows: number, cols: number) => RawWasmWorkspace;

let rawModulePromise: Promise<RawWasmModule> | null = null;
const dynamicImport = new Function("path", "return import(path);") as (path: string) => Promise<RawWasmModule>;
const MODULE_INIT_STATE = new WeakMap<object, { output: unknown; memory: WebAssembly.Memory | null }>();
const WORKSPACE_CTOR_ALIASES = [
  "SweepVelocityWorkspace",
  "SweepWorkspace",
  "OpenDealiasSweepWorkspace",
  "DealiasSweepWorkspace",
] as const;
const WORKSPACE_METHOD_ALIASES = {
  observedPtr: ["observedPtr", "observed_ptr"],
  referencePtr: ["referencePtr", "reference_ptr"],
  previousPtr: ["previousPtr", "previous_ptr"],
  velocityPtr: ["velocityPtr", "velocity_ptr"],
  metadataJson: ["metadataJson", "metadata_json"],
  free: ["free"],
  runZw06VelocityOnly: ["runZw06VelocityOnly", "run_zw06_velocity_only"],
  runRegionGraphVelocityOnly: ["runRegionGraphVelocityOnly", "run_region_graph_velocity_only"],
  runVariationalVelocityOnly: ["runVariationalVelocityOnly", "run_variational_velocity_only"],
  runJH01VelocityOnly: ["runJH01VelocityOnly", "run_jh01_velocity_only"],
} as const;

export interface WasmSweepWorkspacePhaseTimings {
  marshalInMs: number;
  wasmMs: number;
  marshalOutMs: number;
  totalMs: number;
}

export interface OpenDealiasWasmSweepWorkspace {
  readonly rows: number;
  readonly cols: number;
  readonly length: number;
  readonly memory: WebAssembly.Memory;
  readonly observedView: Float32Array;
  readonly velocityView: Float32Array;
  readonly referenceView?: Float32Array;
  readonly previousView?: Float32Array;
  supportsVelocityAlgorithm(algorithm: "zw06" | "region_graph" | "variational" | "jh01"): boolean;
  runZw06Velocity(nyquist: number, reference?: PackedSweep | number[][]): WasmSweepWorkspacePhaseTimings;
  runRegionGraphVelocity(nyquist: number, options?: RegionGraphOptions): WasmSweepWorkspacePhaseTimings;
  runVariationalVelocity(nyquist: number, options?: VariationalOptions): WasmSweepWorkspacePhaseTimings;
  runJH01Velocity(nyquist: number, previousCorrected: PackedSweep | number[][], options?: JH01Options): WasmSweepWorkspacePhaseTimings;
  snapshotVelocity(): Float32Array;
  free(): void;
}

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

function nowMs(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

function findNamedFunction(target: Record<string, unknown>, aliases: readonly string[]): ((...args: unknown[]) => unknown) | null {
  for (const alias of aliases) {
    const candidate = target[alias];
    if (typeof candidate === "function") {
      return candidate as (...args: unknown[]) => unknown;
    }
  }
  return null;
}

function findWorkspaceCtor(module: RawWasmModule): RawWasmWorkspaceCtor | null {
  for (const alias of WORKSPACE_CTOR_ALIASES) {
    const candidate = module[alias];
    if (typeof candidate === "function") {
      return candidate as RawWasmWorkspaceCtor;
    }
  }
  return null;
}

function getWasmMemory(module: RawWasmModule): WebAssembly.Memory | null {
  const direct = module.memory;
  if (direct instanceof WebAssembly.Memory) {
    return direct;
  }
  const state = MODULE_INIT_STATE.get(module);
  if (state?.memory instanceof WebAssembly.Memory) {
    return state.memory;
  }
  return null;
}

function rememberModuleInit(module: RawWasmModule, output: unknown): void {
  const memory =
    typeof output === "object" && output !== null && (output as Record<string, unknown>).memory instanceof WebAssembly.Memory
      ? (output as Record<string, unknown>).memory as WebAssembly.Memory
      : null;
  MODULE_INIT_STATE.set(module, { output, memory });
}

function copyPackedSweepToFloat32(target: Float32Array, sweep: PackedSweep): void {
  for (let index = 0; index < sweep.data.length; index++) {
    target[index] = Math.fround(sweep.data[index]);
  }
}

function fillFloat32(target: Float32Array, value: number): void {
  target.fill(Math.fround(value));
}

function toFloat32View(memory: WebAssembly.Memory, pointer: number, length: number): Float32Array {
  return new Float32Array(memory.buffer, pointer, length);
}

type WorkspaceVelocityRunner = "zw06" | "region_graph" | "variational" | "jh01";

class WasmSweepWorkspaceBridge implements OpenDealiasWasmSweepWorkspace {
  readonly rows: number;
  readonly cols: number;
  readonly length: number;
  readonly memory: WebAssembly.Memory;
  readonly workspace: RawWasmWorkspace;
  private readonly observedPointer: number;
  private readonly referencePointer?: number;
  private readonly previousPointer?: number;
  private readonly velocityPointer: number;
  private readonly runners: Partial<Record<WorkspaceVelocityRunner, (...args: unknown[]) => unknown>>;
  private readonly metadataGetter?: () => unknown;
  observedView: Float32Array;
  velocityView: Float32Array;
  referenceView?: Float32Array;
  previousView?: Float32Array;

  constructor(module: RawWasmModule, rows: number, cols: number) {
    const workspaceCtor = findWorkspaceCtor(module);
    const memory = getWasmMemory(module);
    if (!workspaceCtor || !memory) {
      throw new Error("wasm module does not expose a persistent sweep workspace");
    }
    const length = rows * cols;
    let workspace: RawWasmWorkspace;
    try {
      workspace = new workspaceCtor(rows, cols);
    } catch {
      workspace = new workspaceCtor(length, 1);
    }
    const observedPtr = findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.observedPtr);
    const velocityPtr = findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.velocityPtr);
    if (!observedPtr || !velocityPtr) {
      const free = findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.free);
      free?.call(workspace);
      throw new Error("persistent sweep workspace is missing observed/velocity pointers");
    }

    this.rows = rows;
    this.cols = cols;
    this.length = length;
    this.memory = memory;
    this.workspace = workspace;
    this.observedPointer = Number(observedPtr.call(workspace));
    this.velocityPointer = Number(velocityPtr.call(workspace));

    const referencePtr = findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.referencePtr);
    if (referencePtr) {
      this.referencePointer = Number(referencePtr.call(workspace));
    }
    const previousPtr = findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.previousPtr);
    if (previousPtr) {
      this.previousPointer = Number(previousPtr.call(workspace));
    }

    this.runners = {
      zw06: findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.runZw06VelocityOnly) ?? undefined,
      region_graph: findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.runRegionGraphVelocityOnly) ?? undefined,
      variational: findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.runVariationalVelocityOnly) ?? undefined,
      jh01: findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.runJH01VelocityOnly) ?? undefined,
    };
    this.metadataGetter = findNamedFunction(workspace, WORKSPACE_METHOD_ALIASES.metadataJson) ?? undefined;

    this.observedView = toFloat32View(memory, this.observedPointer, this.length);
    this.velocityView = toFloat32View(memory, this.velocityPointer, this.length);
    if (this.referencePointer !== undefined) {
      this.referenceView = toFloat32View(memory, this.referencePointer, this.length);
    }
    if (this.previousPointer !== undefined) {
      this.previousView = toFloat32View(memory, this.previousPointer, this.length);
    }
  }

  private refreshViews(): void {
    this.observedView = toFloat32View(this.memory, this.observedPointer, this.length);
    this.velocityView = toFloat32View(this.memory, this.velocityPointer, this.length);
    if (this.referencePointer !== undefined) {
      this.referenceView = toFloat32View(this.memory, this.referencePointer, this.length);
    }
    if (this.previousPointer !== undefined) {
      this.previousView = toFloat32View(this.memory, this.previousPointer, this.length);
    }
  }

  workspaceMetadata(): Record<string, unknown> {
    if (!this.metadataGetter) {
      return {};
    }
    try {
      return parseMetadataJson(this.metadataGetter.call(this.workspace));
    } catch {
      return {};
    }
  }

  private timeRun(run: () => void): WasmSweepWorkspacePhaseTimings {
    const t0 = nowMs();
    run();
    const t1 = nowMs();
    this.refreshViews();
    const t2 = nowMs();
    return {
      marshalInMs: Number((t1 - t0).toFixed(2)),
      wasmMs: 0,
      marshalOutMs: Number((t2 - t1).toFixed(2)),
      totalMs: Number((t2 - t0).toFixed(2)),
    };
  }

  private profileCall(
    algorithm: WorkspaceVelocityRunner,
    beforeRun: () => void,
    invoke: (...args: unknown[]) => unknown,
    args: unknown[],
  ): WasmSweepWorkspacePhaseTimings {
    this.refreshViews();
    const t0 = nowMs();
    beforeRun();
    const t1 = nowMs();
    invoke.call(this.workspace, ...args);
    const t2 = nowMs();
    this.refreshViews();
    const t3 = nowMs();
    return {
      marshalInMs: Number((t1 - t0).toFixed(2)),
      wasmMs: Number((t2 - t1).toFixed(2)),
      marshalOutMs: Number((t3 - t2).toFixed(2)),
      totalMs: Number((t3 - t0).toFixed(2)),
    };
  }

  supportsVelocityAlgorithm(algorithm: WorkspaceVelocityRunner): boolean {
    return typeof this.runners[algorithm] === "function";
  }

  runZw06Velocity(nyquist: number, reference?: PackedSweep | number[][]): WasmSweepWorkspacePhaseTimings {
    const runner = this.runners.zw06;
    if (!runner) {
      throw new Error("persistent sweep workspace does not support zw06 velocity");
    }
    const packedReference = reference ? packSweep(reference) : undefined;
    if (packedReference && !this.referenceView) {
      throw new Error("persistent sweep workspace does not expose a reference buffer");
    }
    return this.profileCall(
      "zw06",
      () => {
        if (packedReference && this.referenceView) {
          copyPackedSweepToFloat32(this.referenceView, packedReference);
        } else if (this.referenceView) {
          fillFloat32(this.referenceView, Number.NaN);
        }
      },
      runner,
      [nyquist, 0.35, true, 12, true, true],
    );
  }

  runRegionGraphVelocity(nyquist: number, options?: RegionGraphOptions): WasmSweepWorkspacePhaseTimings {
    const runner = this.runners.region_graph;
    if (!runner) {
      throw new Error("persistent sweep workspace does not support region_graph velocity");
    }
    const normalized = normalizeRegionGraphOptions(options);
    const packedReference = normalized?.reference ? packSweep(normalized.reference) : undefined;
    if (packedReference && !this.referenceView) {
      throw new Error("persistent sweep workspace does not expose a reference buffer");
    }
    return this.profileCall(
      "region_graph",
      () => {
        if (packedReference && this.referenceView) {
          copyPackedSweepToFloat32(this.referenceView, packedReference);
        } else if (this.referenceView) {
          fillFloat32(this.referenceView, Number.NaN);
        }
      },
      runner,
      [
        nyquist,
        normalized?.blockRows,
        normalized?.blockCols,
        normalized?.referenceWeight ?? 0.75,
        normalized?.maxIterations ?? 6,
        normalized?.maxAbsFold ?? 8,
        normalized?.wrapAzimuth ?? true,
        normalized?.minRegionArea ?? 4,
        normalized?.minValidFraction ?? 0.15,
      ],
    );
  }

  runVariationalVelocity(nyquist: number, options?: VariationalOptions): WasmSweepWorkspacePhaseTimings {
    const runner = this.runners.variational;
    if (!runner) {
      throw new Error("persistent sweep workspace does not support variational velocity");
    }
    const normalized = normalizeVariationalOptions(options);
    if (normalized.bootstrapMethod === "region_graph") {
      throw new Error("persistent sweep workspace variational path only supports zw06 bootstrap");
    }
    const packedReference = normalized.reference ? packSweep(normalized.reference) : undefined;
    if (packedReference && !this.referenceView) {
      throw new Error("persistent sweep workspace does not expose a reference buffer");
    }
    return this.profileCall(
      "variational",
      () => {
        if (packedReference && this.referenceView) {
          copyPackedSweepToFloat32(this.referenceView, packedReference);
        } else if (this.referenceView) {
          fillFloat32(this.referenceView, Number.NaN);
        }
      },
      runner,
      [
        nyquist,
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
      ],
    );
  }

  runJH01Velocity(nyquist: number, previousCorrected: PackedSweep | number[][], options?: JH01Options): WasmSweepWorkspacePhaseTimings {
    const runner = this.runners.jh01;
    if (!runner) {
      throw new Error("persistent sweep workspace does not support jh01 velocity");
    }
    const packedPrevious = packSweep(previousCorrected);
    if (!this.previousView) {
      throw new Error("persistent sweep workspace does not expose a previous-sweep buffer");
    }
    return this.profileCall(
      "jh01",
      () => {
        copyPackedSweepToFloat32(this.previousView!, packedPrevious);
      },
      runner,
      [nyquist, options?.shiftAz ?? 0, options?.shiftRange ?? 0],
    );
  }

  snapshotVelocity(): Float32Array {
    this.refreshViews();
    return this.velocityView.slice();
  }

  free(): void {
    const free = findNamedFunction(this.workspace, WORKSPACE_METHOD_ALIASES.free);
    free?.call(this.workspace);
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

function workspaceVelocityResult(
  workspace: WasmSweepWorkspaceBridge,
  observed: PackedSweep,
  method: string,
  timings: WasmSweepWorkspacePhaseTimings,
): PackedVelocityResult {
  return {
    velocity: workspace.snapshotVelocity(),
    azimuthCount: observed.azimuthCount,
    gateCount: observed.gateCount,
    metadata: {
      ...workspace.workspaceMetadata(),
      method,
      backend: "open-dealias-wasm",
      path: "workspace",
      timings,
    },
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

export function supportsOpenDealiasWasmSweepWorkspace(module: RawWasmModule): boolean {
  return findWorkspaceCtor(module) !== null && getWasmMemory(module) !== null;
}

export function createOpenDealiasWasmSweepWorkspaceFromModule(
  module: RawWasmModule,
  rows: number,
  cols: number,
): OpenDealiasWasmSweepWorkspace | null {
  if (!supportsOpenDealiasWasmSweepWorkspace(module)) {
    return null;
  }
  return new WasmSweepWorkspaceBridge(module, rows, cols);
}

export async function createOpenDealiasWasmSweepWorkspace(
  rows: number,
  cols: number,
): Promise<OpenDealiasWasmSweepWorkspace | null> {
  const module = await loadRawModule();
  await init();
  return createOpenDealiasWasmSweepWorkspaceFromModule(module, rows, cols);
}

export default async function init(input?: unknown): Promise<unknown> {
  const module = await loadRawModule();
  const initFn = requireFunction<(arg?: unknown) => Promise<unknown> | unknown>(module, "default");
  const output = await initFn(input);
  rememberModuleInit(module, output);
  return output;
}

export function registerOpenDealiasWasmInitOutput(module: RawWasmModule, output: unknown): void {
  rememberModuleInit(module, output);
}

export async function createOpenDealiasBackend(): Promise<OpenDealiasBackend> {
  const module = await loadRawModule();
  await init();
  return createOpenDealiasBackendFromModule(module);
}

export function createOpenDealiasBackendFromModule(module: RawWasmModule): OpenDealiasBackend {
  const call = <T>(name: string, ...args: unknown[]): T =>
    requireFunction<(...callArgs: unknown[]) => T>(module, name)(...args);
  const workspaceCache = new Map<string, OpenDealiasWasmSweepWorkspace>();

  const getWorkspace = (observed: PackedSweep): OpenDealiasWasmSweepWorkspace | null => {
    const key = `${observed.azimuthCount}x${observed.gateCount}`;
    const existing = workspaceCache.get(key);
    if (existing) {
      return existing;
    }
    const created = createOpenDealiasWasmSweepWorkspaceFromModule(module, observed.azimuthCount, observed.gateCount);
    if (created) {
      workspaceCache.set(key, created);
    }
    return created;
  };

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
      const workspace = getWorkspace(observed);
      if (workspace?.supportsVelocityAlgorithm("zw06")) {
        copyPackedSweepToFloat32(workspace.observedView, observed);
        const timings = workspace.runZw06Velocity(nyquist, reference);
        return workspaceVelocityResult(workspace as WasmSweepWorkspaceBridge, observed, "2d_multipass", timings);
      }
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
      const workspace = getWorkspace(observed);
      if (workspace?.supportsVelocityAlgorithm("jh01")) {
        copyPackedSweepToFloat32(workspace.observedView, observed);
        const timings = workspace.runJH01Velocity(nyquist, previousCorrected, options);
        return {
          velocity: workspace.snapshotVelocity(),
          azimuthCount: observed.azimuthCount,
          gateCount: observed.gateCount,
          metadata: {
            method: "temporal_reference",
            backend: "open-dealias-wasm",
            path: "workspace",
            timings,
          },
        };
      }
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
      const workspace = getWorkspace(observed);
      if (workspace?.supportsVelocityAlgorithm("region_graph")) {
        copyPackedSweepToFloat32(workspace.observedView, observed);
        const timings = workspace.runRegionGraphVelocity(nyquist, options);
        return workspaceVelocityResult(workspace as WasmSweepWorkspaceBridge, observed, "region_graph_sweep", timings);
      }
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
      const normalized = normalizeVariationalOptions(options);
      const workspace = getWorkspace(observed);
      if (workspace?.supportsVelocityAlgorithm("variational") && normalized.bootstrapMethod !== "region_graph") {
        copyPackedSweepToFloat32(workspace.observedView, observed);
        const timings = workspace.runVariationalVelocity(nyquist, normalized);
        return workspaceVelocityResult(workspace as WasmSweepWorkspaceBridge, observed, "variational_refine", timings);
      }
      return callVariationalVelocity(observed, nyquist, normalized);
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
