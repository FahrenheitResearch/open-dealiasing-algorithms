export interface DealiasResult {
  velocity: number[][];
  folds: number[][];
  confidence: number[][];
  metadata: Record<string, unknown>;
}

export interface VolumeDealiasResult {
  velocity: number[][][];
  folds: number[][][];
  confidence: number[][][];
  metadata: Record<string, unknown>;
}

export interface PackedSweep {
  data: Float64Array;
  azimuthCount: number;
  gateCount: number;
}

export interface PackedVolume {
  data: Float64Array;
  sweepCount: number;
  azimuthCount: number;
  gateCount: number;
}

export interface SweepPackingOptions {
  noDataValue?: number;
  noDataValues?: ArrayLike<number>;
  zeroIsMissing?: boolean;
}

export interface PackedDealiasResult {
  velocity: Float64Array;
  folds: Int16Array;
  confidence: Float32Array;
  azimuthCount: number;
  gateCount: number;
  metadata: Record<string, unknown>;
}

export interface PackedVelocityResult {
  velocity: Float32Array;
  azimuthCount: number;
  gateCount: number;
  metadata: Record<string, unknown>;
}

export interface PackedVolumeDealiasResult {
  velocity: Float64Array;
  folds: Int16Array;
  confidence: Float32Array;
  sweepCount: number;
  azimuthCount: number;
  gateCount: number;
  metadata: Record<string, unknown>;
}

export interface Xu11Options {
  azimuthDeg: ArrayLike<number>;
  elevationDeg?: number;
  reference?: PackedSweep | number[][];
  backgroundUV?: readonly [number, number];
}

export interface JH01Options {
  shiftAz?: number;
  shiftRange?: number;
}

export interface RegionGraphOptions {
  reference?: PackedSweep | number[][];
  blockRows?: number;
  blockCols?: number;
  referenceWeight?: number;
  maxIterations?: number;
  maxAbsFold?: number;
  wrapAzimuth?: boolean;
  minRegionArea?: number;
  minValidFraction?: number;
}

export interface RecursiveOptions extends RegionGraphOptions {}

export interface VariationalOptions extends RegionGraphOptions {}

export interface MlOptions extends RegionGraphOptions {}

export interface DualPrfOptions {
  maxAbsFold?: number;
  reference?: PackedSweep | number[][];
}

export interface VolumeOptions {
  referenceVolume?: PackedVolume | number[][][];
  previousVolume?: PackedVolume | number[][][];
}

export interface OpenDealiasBackend {
  readonly name: string;
  readonly kind?: "js" | "wasm" | "custom";
  wrapToNyquist?(value: number, nyquist: number): number;
  unfoldToReference?(observed: number, reference: number, nyquist: number): number;
  foldCount?(unfolded: number, observed: number, nyquist: number): number;
  dealiasRadialES90?(observed: ArrayLike<number>, nyquist: number, reference?: ArrayLike<number>): {
    velocity: ArrayLike<number>;
    folds: ArrayLike<number>;
    confidence: ArrayLike<number>;
    metadata?: Record<string, unknown>;
  };
  estimateUniformWindVAD?(observed: PackedSweep, azimuthDeg: ArrayLike<number>, elevationDeg?: number): readonly [number, number];
  dealiasSweepZW06Packed?(observed: PackedSweep, nyquist: number, reference?: PackedSweep): PackedDealiasResult;
  dealiasSweepZW06VelocityPacked?(observed: PackedSweep, nyquist: number, reference?: PackedSweep): PackedVelocityResult;
  dealiasSweepXu11Packed?(observed: PackedSweep, nyquist: number, options: Xu11Options): PackedDealiasResult;
  dealiasSweepXu11VelocityPacked?(observed: PackedSweep, nyquist: number, options: Xu11Options): PackedVelocityResult;
  dealiasSweepJH01Packed?(observed: PackedSweep, nyquist: number, previousCorrected: PackedSweep, options?: JH01Options): PackedDealiasResult;
  dealiasSweepJH01VelocityPacked?(observed: PackedSweep, nyquist: number, previousCorrected: PackedSweep, options?: JH01Options): PackedVelocityResult;
  dealiasSweepRegionGraphPacked?(observed: PackedSweep, nyquist: number, options?: RegionGraphOptions): PackedDealiasResult;
  dealiasSweepRegionGraphVelocityPacked?(observed: PackedSweep, nyquist: number, options?: RegionGraphOptions): PackedVelocityResult;
  dealiasSweepRecursivePacked?(observed: PackedSweep, nyquist: number, options?: RecursiveOptions): PackedDealiasResult;
  dealiasSweepVariationalPacked?(observed: PackedSweep, nyquist: number, options?: VariationalOptions): PackedDealiasResult;
  dealiasSweepVariationalVelocityPacked?(observed: PackedSweep, nyquist: number, options?: VariationalOptions): PackedVelocityResult;
  dealiasSweepMLPacked?(observed: PackedSweep, nyquist: number, options?: MlOptions): PackedDealiasResult;
  dealiasDualPrfPacked?(lowObserved: PackedSweep, highObserved: PackedSweep, lowNyquist: number, highNyquist: number, options?: DualPrfOptions): PackedDealiasResult;
  dealiasVolumeJH01Packed?(observed: PackedVolume, nyquist: number | ArrayLike<number>, previousCorrected: PackedVolume, options?: VolumeOptions): PackedVolumeDealiasResult;
  dealiasVolume3DPacked?(observed: PackedVolume, nyquist: number | ArrayLike<number>, options?: VolumeOptions): PackedVolumeDealiasResult;
}

export type OpenDealiasBackendSource =
  | OpenDealiasBackend
  | Record<string, unknown>
  | Promise<OpenDealiasBackend | Record<string, unknown>>
  | (() => OpenDealiasBackend | Record<string, unknown> | Promise<OpenDealiasBackend | Record<string, unknown>>);

const WASM_METHOD_ALIASES = {
  dealiasSweepZW06Packed: ["dealiasSweepZW06Packed", "dealias_sweep_zw06_packed", "dealiasSweepZw06Packed"],
  dealiasSweepZW06VelocityPacked: ["dealiasSweepZW06VelocityPacked", "dealiasSweepZw06Velocity"],
  dealiasSweepXu11Packed: ["dealiasSweepXu11Packed", "dealias_sweep_xu11_packed", "dealiasSweepXu11"],
  dealiasSweepXu11VelocityPacked: ["dealiasSweepXu11VelocityPacked", "dealiasSweepXu11Velocity"],
  dealiasSweepJH01Packed: ["dealiasSweepJH01Packed", "dealias_sweep_jh01_packed", "dealiasSweepJh01Packed"],
  dealiasSweepJH01VelocityPacked: ["dealiasSweepJH01VelocityPacked", "dealiasSweepJh01Velocity"],
  dealiasSweepRegionGraphPacked: ["dealiasSweepRegionGraphPacked", "dealias_sweep_region_graph_packed"],
  dealiasSweepRegionGraphVelocityPacked: ["dealiasSweepRegionGraphVelocityPacked", "dealiasSweepRegionGraphVelocity"],
  dealiasSweepRecursivePacked: ["dealiasSweepRecursivePacked", "dealias_sweep_recursive_packed"],
  dealiasSweepVariationalPacked: ["dealiasSweepVariationalPacked", "dealias_sweep_variational_packed", "dealiasSweepVariationalRefinePacked"],
  dealiasSweepVariationalVelocityPacked: ["dealiasSweepVariationalVelocityPacked", "dealiasSweepVariationalVelocity"],
  dealiasSweepMLPacked: ["dealiasSweepMLPacked", "dealias_sweep_ml_packed"],
  dealiasDualPrfPacked: ["dealiasDualPrfPacked", "dealias_dual_prf_packed", "dealiasDualPRFPacked"],
  dealiasVolumeJH01Packed: ["dealiasVolumeJH01Packed", "dealias_volume_jh01_packed", "dealiasVolumeJh01Packed"],
  dealiasVolume3DPacked: ["dealiasVolume3DPacked", "dealias_volume_3d_packed", "dealiasVolume3dPacked"],
  estimateUniformWindVAD: ["estimateUniformWindVAD", "estimate_uniform_wind_vad"],
  wrapToNyquist: ["wrapToNyquist", "wrap_to_nyquist"],
  unfoldToReference: ["unfoldToReference", "unfold_to_reference"],
  foldCount: ["foldCount", "fold_count"],
} as const;

let activeBackend: OpenDealiasBackend | null = null;

function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

function ensurePositiveNyquist(nyquist: number): void {
  if (!(nyquist > 0)) {
    throw new Error("nyquist must be positive");
  }
}

function medianFinite(values: number[]): number {
  const finite = values.filter(isFiniteNumber).sort((a, b) => a - b);
  if (finite.length === 0) return Number.NaN;
  const mid = Math.floor(finite.length / 2);
  return finite.length % 2 ? finite[mid] : 0.5 * (finite[mid - 1] + finite[mid]);
}

function isArrayOfArrays(value: unknown): value is number[][] {
  return Array.isArray(value) && (value.length === 0 || Array.isArray(value[0]));
}

function isArrayOfArrayOfArrays(value: unknown): value is number[][][] {
  return Array.isArray(value) && (value.length === 0 || isArrayOfArrays(value[0]));
}

function hasShapeCounts(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

export function isPackedSweep(value: unknown): value is PackedSweep {
  return hasShapeCounts(value)
    && value.data instanceof Float64Array
    && typeof value.azimuthCount === "number"
    && typeof value.gateCount === "number";
}

export function isPackedVolume(value: unknown): value is PackedVolume {
  return hasShapeCounts(value)
    && value.data instanceof Float64Array
    && typeof value.sweepCount === "number"
    && typeof value.azimuthCount === "number"
    && typeof value.gateCount === "number";
}

function cloneMatrix(field: number[][]): number[][] {
  return field.map((row) => row.slice());
}

function cloneVolume(field: number[][][]): number[][][] {
  return field.map((sweep) => cloneMatrix(sweep));
}

function buildNoDataMatcher(options?: SweepPackingOptions): ((value: number) => boolean) | null {
  if (!options) {
    return null;
  }
  const sentinels = new Set<number>();
  if (typeof options.noDataValue === "number" && Number.isFinite(options.noDataValue)) {
    sentinels.add(options.noDataValue);
  }
  if (options.noDataValues) {
    for (const value of Array.from(options.noDataValues)) {
      if (typeof value === "number" && Number.isFinite(value)) {
        sentinels.add(value);
      }
    }
  }
  const zeroIsMissing = options.zeroIsMissing === true;
  if (!zeroIsMissing && sentinels.size === 0) {
    return null;
  }
  return (value: number): boolean => {
    if (!Number.isFinite(value)) {
      return true;
    }
    if (zeroIsMissing && value === 0) {
      return true;
    }
    return sentinels.has(value);
  };
}

export function sanitizePackedSweep(field: PackedSweep, options?: SweepPackingOptions): PackedSweep {
  const isMissing = buildNoDataMatcher(options);
  if (!isMissing) {
    return field;
  }
  const data = new Float64Array(field.data.length);
  let changed = false;
  for (let index = 0; index < field.data.length; index++) {
    const value = field.data[index];
    const normalized = isMissing(value) ? Number.NaN : value;
    data[index] = normalized;
    changed ||= !Object.is(normalized, value);
  }
  if (!changed) {
    return field;
  }
  return { data, azimuthCount: field.azimuthCount, gateCount: field.gateCount };
}

export function packSweep(field: number[][] | PackedSweep, options?: SweepPackingOptions): PackedSweep {
  const isMissing = buildNoDataMatcher(options);
  if (isPackedSweep(field)) {
    return sanitizePackedSweep(field, options);
  }
  if (!isArrayOfArrays(field) || field.length === 0 || field[0].length === 0) {
    throw new Error("sweep must be a non-empty 2D numeric matrix");
  }
  const azimuthCount = field.length;
  const gateCount = field[0].length;
  const data = new Float64Array(azimuthCount * gateCount);
  for (let i = 0; i < azimuthCount; i++) {
    if (field[i].length !== gateCount) {
      throw new Error("sweep rows must all have the same length");
    }
    for (let j = 0; j < gateCount; j++) {
      const value = field[i][j];
      data[i * gateCount + j] = isMissing?.(value) ? Number.NaN : value;
    }
  }
  return { data, azimuthCount, gateCount };
}

export function unpackSweep(field: PackedSweep): number[][] {
  const out = Array.from({ length: field.azimuthCount }, () => new Array<number>(field.gateCount).fill(Number.NaN));
  for (let i = 0; i < field.azimuthCount; i++) {
    for (let j = 0; j < field.gateCount; j++) {
      out[i][j] = field.data[i * field.gateCount + j];
    }
  }
  return out;
}

export function packVolume(field: number[][][] | PackedVolume): PackedVolume {
  if (isPackedVolume(field)) {
    return field;
  }
  if (!isArrayOfArrayOfArrays(field) || field.length === 0 || field[0].length === 0 || field[0][0].length === 0) {
    throw new Error("volume must be a non-empty 3D numeric array");
  }
  const sweepCount = field.length;
  const azimuthCount = field[0].length;
  const gateCount = field[0][0].length;
  const data = new Float64Array(sweepCount * azimuthCount * gateCount);
  for (let s = 0; s < sweepCount; s++) {
    if (field[s].length !== azimuthCount) {
      throw new Error("volume sweeps must all have the same azimuth count");
    }
    for (let i = 0; i < azimuthCount; i++) {
      if (field[s][i].length !== gateCount) {
        throw new Error("volume rows must all have the same gate count");
      }
      for (let j = 0; j < gateCount; j++) {
        data[s * azimuthCount * gateCount + i * gateCount + j] = field[s][i][j];
      }
    }
  }
  return { data, sweepCount, azimuthCount, gateCount };
}

export function unpackVolume(field: PackedVolume): number[][][] {
  const out = Array.from({ length: field.sweepCount }, () => Array.from({ length: field.azimuthCount }, () => new Array<number>(field.gateCount).fill(Number.NaN)));
  for (let s = 0; s < field.sweepCount; s++) {
    for (let i = 0; i < field.azimuthCount; i++) {
      for (let j = 0; j < field.gateCount; j++) {
        out[s][i][j] = field.data[s * field.azimuthCount * field.gateCount + i * field.gateCount + j];
      }
    }
  }
  return out;
}

function normalizePackedResult(value: PackedDealiasResult | Record<string, unknown>, azimuthCount: number, gateCount: number): PackedDealiasResult {
  const velocity = value.velocity instanceof Float64Array ? value.velocity : new Float64Array(value.velocity as ArrayLike<number>);
  const folds = value.folds instanceof Int16Array ? value.folds : Int16Array.from(value.folds as ArrayLike<number>);
  const confidence = value.confidence instanceof Float32Array ? value.confidence : Float32Array.from(value.confidence as ArrayLike<number>);
  return {
    velocity,
    folds,
    confidence,
    azimuthCount,
    gateCount,
    metadata: typeof value.metadata === "object" && value.metadata !== null ? value.metadata as Record<string, unknown> : {},
  };
}

function normalizePackedVelocityResult(value: PackedVelocityResult | Record<string, unknown>, azimuthCount: number, gateCount: number): PackedVelocityResult {
  const velocity = value.velocity instanceof Float32Array ? value.velocity : Float32Array.from(value.velocity as ArrayLike<number>);
  return {
    velocity,
    azimuthCount,
    gateCount,
    metadata: typeof value.metadata === "object" && value.metadata !== null ? value.metadata as Record<string, unknown> : {},
  };
}

function normalizePackedVolumeResult(value: PackedVolumeDealiasResult | Record<string, unknown>, sweepCount: number, azimuthCount: number, gateCount: number): PackedVolumeDealiasResult {
  const velocity = value.velocity instanceof Float64Array ? value.velocity : new Float64Array(value.velocity as ArrayLike<number>);
  const folds = value.folds instanceof Int16Array ? value.folds : Int16Array.from(value.folds as ArrayLike<number>);
  const confidence = value.confidence instanceof Float32Array ? value.confidence : Float32Array.from(value.confidence as ArrayLike<number>);
  return {
    velocity,
    folds,
    confidence,
    sweepCount,
    azimuthCount,
    gateCount,
    metadata: typeof value.metadata === "object" && value.metadata !== null ? value.metadata as Record<string, unknown> : {},
  };
}

export function unpackPackedResult(result: PackedDealiasResult): DealiasResult {
  return {
    velocity: unpackSweep({ data: result.velocity, azimuthCount: result.azimuthCount, gateCount: result.gateCount }),
    folds: unpackSweep({ data: Float64Array.from(result.folds), azimuthCount: result.azimuthCount, gateCount: result.gateCount }).map((row) => row.map((v) => Math.trunc(v))),
    confidence: unpackSweep({ data: Float64Array.from(result.confidence), azimuthCount: result.azimuthCount, gateCount: result.gateCount }),
    metadata: { ...result.metadata },
  };
}

export function unpackPackedVelocityResult(result: PackedVelocityResult): number[][] {
  return unpackSweep({ data: Float64Array.from(result.velocity), azimuthCount: result.azimuthCount, gateCount: result.gateCount });
}

export function unpackPackedVolumeResult(result: PackedVolumeDealiasResult): VolumeDealiasResult {
  const velocity = unpackVolume({ data: result.velocity, sweepCount: result.sweepCount, azimuthCount: result.azimuthCount, gateCount: result.gateCount });
  const foldsRaw = unpackVolume({ data: Float64Array.from(result.folds), sweepCount: result.sweepCount, azimuthCount: result.azimuthCount, gateCount: result.gateCount });
  const confidence = unpackVolume({ data: Float64Array.from(result.confidence), sweepCount: result.sweepCount, azimuthCount: result.azimuthCount, gateCount: result.gateCount });
  return {
    velocity,
    folds: foldsRaw.map((sweep) => sweep.map((row) => row.map((v) => Math.trunc(v)))),
    confidence,
    metadata: { ...result.metadata },
  };
}

function packReference(reference: PackedSweep | number[][] | undefined): PackedSweep | undefined {
  if (reference === undefined) return undefined;
  return packSweep(reference);
}

function packVolumeReference(reference: PackedVolume | number[][][] | undefined): PackedVolume | undefined {
  if (reference === undefined) return undefined;
  return packVolume(reference);
}

function packedResultFromMatrix(result: DealiasResult): PackedDealiasResult {
  return {
    velocity: packSweep(result.velocity).data,
    folds: Int16Array.from(packSweep(result.folds).data, (value) => Math.trunc(value)),
    confidence: Float32Array.from(packSweep(result.confidence).data),
    azimuthCount: result.velocity.length,
    gateCount: result.velocity[0]?.length ?? 0,
    metadata: { ...result.metadata },
  };
}

function packedVolumeResultFromMatrix(result: VolumeDealiasResult): PackedVolumeDealiasResult {
  const velocity = packVolume(result.velocity);
  const folds = packVolume(result.folds);
  const confidence = packVolume(result.confidence);
  return {
    velocity: velocity.data,
    folds: Int16Array.from(folds.data, (value) => Math.trunc(value)),
    confidence: Float32Array.from(confidence.data),
    sweepCount: velocity.sweepCount,
    azimuthCount: velocity.azimuthCount,
    gateCount: velocity.gateCount,
    metadata: { ...result.metadata },
  };
}

export function wrapToNyquist(value: number, nyquist: number): number {
  ensurePositiveNyquist(nyquist);
  if (!isFiniteNumber(value)) return Number.NaN;
  return ((value + nyquist) % (2 * nyquist) + 2 * nyquist) % (2 * nyquist) - nyquist;
}

export function unfoldToReference(observed: number, reference: number, nyquist: number): number {
  ensurePositiveNyquist(nyquist);
  if (!isFiniteNumber(observed) || !isFiniteNumber(reference)) return Number.NaN;
  const fold = Math.round((reference - observed) / (2 * nyquist));
  return observed + 2 * nyquist * fold;
}

export function foldCount(unfolded: number, observed: number, nyquist: number): number {
  ensurePositiveNyquist(nyquist);
  if (!isFiniteNumber(unfolded) || !isFiniteNumber(observed)) return 0;
  return Math.round((unfolded - observed) / (2 * nyquist));
}

function shift2D(field: number[][], da: number, dr: number): number[][] {
  const nAz = field.length;
  const nR = field[0].length;
  const out = Array.from({ length: nAz }, () => new Array<number>(nR).fill(Number.NaN));
  for (let i = 0; i < nAz; i++) {
    for (let j = 0; j < nR; j++) {
      let srcI = i - da;
      srcI = ((srcI % nAz) + nAz) % nAz;
      const srcJ = j - dr;
      if (srcJ < 0 || srcJ >= nR) continue;
      out[i][j] = field[srcI][srcJ];
    }
  }
  return out;
}

function neighborValues(field: number[][], i: number, j: number): number[] {
  const nAz = field.length;
  const nR = field[0].length;
  const values: number[] = [];
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      if (di === 0 && dj === 0) continue;
      const ii = (i + di + nAz) % nAz;
      const jj = j + dj;
      if (jj < 0 || jj >= nR) continue;
      const value = field[ii][jj];
      if (isFiniteNumber(value)) values.push(value);
    }
  }
  return values;
}

export function buildReferenceFromUV(azimuthDeg: ArrayLike<number>, nRange: number, u: number, v: number, elevationDeg = 0): number[][] {
  const el = elevationDeg * Math.PI / 180;
  return Array.from({ length: azimuthDeg.length }, (_, i) => {
    const az = azimuthDeg[i] * Math.PI / 180;
    const vr = Math.cos(el) * (u * Math.sin(az) + v * Math.cos(az));
    return new Array<number>(nRange).fill(vr);
  });
}

export function buildPackedReferenceFromUV(azimuthDeg: ArrayLike<number>, nRange: number, u: number, v: number, elevationDeg = 0): PackedSweep {
  return packSweep(buildReferenceFromUV(azimuthDeg, nRange, u, v, elevationDeg));
}

export function estimateUniformWindVAD(observed: number[][] | PackedSweep, azimuthDeg: ArrayLike<number>, elevationDeg = 0): readonly [number, number] {
  const sweep = isPackedSweep(observed) ? unpackSweep(observed) : observed;
  const backend = getOpenDealiasBackend();
  if (backend.estimateUniformWindVAD) {
    return backend.estimateUniformWindVAD(packSweep(sweep), azimuthDeg, elevationDeg);
  }
  const elFactor = Math.max(Math.cos(elevationDeg * Math.PI / 180), 1e-6);
  let s11 = 0;
  let s22 = 0;
  let s12 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < sweep.length; i++) {
    const az = azimuthDeg[i] * Math.PI / 180;
    const x1 = Math.sin(az) * elFactor;
    const x2 = Math.cos(az) * elFactor;
    for (let j = 0; j < sweep[i].length; j++) {
      const vr = sweep[i][j];
      if (!isFiniteNumber(vr)) continue;
      s11 += x1 * x1;
      s22 += x2 * x2;
      s12 += x1 * x2;
      b1 += x1 * vr;
      b2 += x2 * vr;
    }
  }
  const det = s11 * s22 - s12 * s12;
  if (!(Math.abs(det) > 1e-9)) {
    return [0, 0];
  }
  const u = (b1 * s22 - b2 * s12) / det;
  const v = (b2 * s11 - b1 * s12) / det;
  return [u, v];
}

function makeMatrixResult(velocity: number[][], folds: number[][], confidence: number[][], metadata: Record<string, unknown>): DealiasResult {
  return { velocity, folds, confidence, metadata };
}

function makeEmptySweepResult(observed: number[][], method: string, metadata: Record<string, unknown> = {}): DealiasResult {
  return makeMatrixResult(
    cloneMatrix(observed),
    observed.map((row) => new Array<number>(row.length).fill(0)),
    observed.map((row) => new Array<number>(row.length).fill(0)),
    { method, ...metadata },
  );
}

function dealiasRadialES90Js(observed: ArrayLike<number>, nyquist: number, reference?: ArrayLike<number>): {
  velocity: number[];
  folds: number[];
  confidence: number[];
  metadata: Record<string, unknown>;
} {
  ensurePositiveNyquist(nyquist);
  const n = observed.length;
  const corrected = new Array<number>(n).fill(Number.NaN);
  const confidence = new Array<number>(n).fill(0);
  let seed = -1;
  for (let i = Math.floor(n / 2); i < n; i++) {
    if (isFiniteNumber(observed[i])) { seed = i; break; }
  }
  if (seed < 0) {
    for (let i = Math.floor(n / 2) - 1; i >= 0; i--) {
      if (isFiniteNumber(observed[i])) { seed = i; break; }
    }
  }
  if (seed < 0) {
    return { velocity: corrected, folds: new Array<number>(n).fill(0), confidence, metadata: { method: "es90" } };
  }

  const seedRef = reference && isFiniteNumber(reference[seed]) ? reference[seed] : Number.NaN;
  corrected[seed] = isFiniteNumber(seedRef) ? unfoldToReference(observed[seed], seedRef, nyquist) : observed[seed];
  confidence[seed] = isFiniteNumber(seedRef) ? 0.98 : 0.8;

  for (const direction of [1, -1]) {
    let last = seed;
    for (let idx = seed + direction; idx >= 0 && idx < n; idx += direction) {
      if (!isFiniteNumber(observed[idx])) continue;
      const refs: number[] = [];
      if (isFiniteNumber(corrected[last])) refs.push(corrected[last]);
      if (reference && isFiniteNumber(reference[idx])) refs.push(reference[idx]);
      if (refs.length === 0) {
        corrected[idx] = observed[idx];
        confidence[idx] = 0.15;
      } else {
        const localRef = medianFinite(refs);
        corrected[idx] = unfoldToReference(observed[idx], localRef, nyquist);
        const mismatch = Math.abs(corrected[idx] - localRef);
        confidence[idx] = Math.exp(-0.5 * (mismatch / (0.45 * nyquist)) ** 2);
      }
      last = idx;
    }
  }

  return {
    velocity: corrected,
    folds: corrected.map((value, i) => foldCount(value, observed[i], nyquist)),
    confidence,
    metadata: { method: "es90" },
  };
}

function dealiasSweepZW06Js(observed: number[][], nyquist: number, reference?: number[][]): DealiasResult {
  ensurePositiveNyquist(nyquist);
  if (observed.length === 0 || observed[0].length === 0) {
    return makeEmptySweepResult(observed, "zw06");
  }
  const nAz = observed.length;
  const nR = observed[0].length;
  const corrected = Array.from({ length: nAz }, () => new Array<number>(nR).fill(Number.NaN));
  const confidence = Array.from({ length: nAz }, () => new Array<number>(nR).fill(0));
  const passes = [
    { minNeighbors: 3, maxMismatch: 0.35 * nyquist, referenceOnly: false },
    { minNeighbors: 2, maxMismatch: 0.55 * nyquist, referenceOnly: true },
    { minNeighbors: 1, maxMismatch: 0.85 * nyquist, referenceOnly: true },
  ];

  for (let i = 0; i < nAz; i++) {
    for (let j = 0; j < nR; j++) {
      const obs = observed[i][j];
      if (!isFiniteNumber(obs)) continue;
      if (reference && isFiniteNumber(reference[i][j])) {
        const candidate = unfoldToReference(obs, reference[i][j], nyquist);
        if (Math.abs(candidate - reference[i][j]) <= 0.65 * nyquist) {
          corrected[i][j] = candidate;
          confidence[i][j] = 0.9;
        }
      } else if (Math.abs(obs) <= 0.35 * nyquist) {
        corrected[i][j] = obs;
        confidence[i][j] = 0.72;
      }
    }
  }

  for (const pass of passes) {
    let changed = true;
    let guard = 0;
    while (changed && guard < 12) {
      guard += 1;
      changed = false;
      for (let i = 0; i < nAz; i++) {
        for (let j = 0; j < nR; j++) {
          if (!isFiniteNumber(observed[i][j]) || isFiniteNumber(corrected[i][j])) continue;
          const refs = neighborValues(corrected, i, j);
          if (reference && isFiniteNumber(reference[i][j])) refs.push(reference[i][j]);
          const enough = refs.length >= pass.minNeighbors || (pass.referenceOnly && reference && isFiniteNumber(reference[i][j]));
          if (!enough) continue;
          const ref = medianFinite(refs);
          const candidate = unfoldToReference(observed[i][j], ref, nyquist);
          const mismatch = Math.abs(candidate - ref);
          if (mismatch <= pass.maxMismatch) {
            corrected[i][j] = candidate;
            confidence[i][j] = Math.exp(-0.5 * (mismatch / (0.4 * nyquist)) ** 2);
            changed = true;
          }
        }
      }
    }
  }

  const folds = observed.map((row, i) => row.map((value, j) => foldCount(corrected[i][j], value, nyquist)));
  return makeMatrixResult(corrected, folds, confidence, { method: "zw06" });
}

function dealiasSweepXu11Js(observed: number[][], nyquist: number, options: Xu11Options): DealiasResult {
  const reference = options.reference
    ? (isPackedSweep(options.reference) ? unpackSweep(options.reference) : options.reference)
    : buildReferenceFromUV(
      options.azimuthDeg,
      observed[0]?.length ?? 0,
      ...(options.backgroundUV ?? estimateUniformWindVAD(observed, options.azimuthDeg, options.elevationDeg ?? 0)),
      options.elevationDeg ?? 0,
    );
  const result = dealiasSweepZW06Js(observed, nyquist, reference);
  return { ...result, metadata: { ...result.metadata, method: "xu11", anchor: "vad" } };
}

function dealiasSweepJH01Js(observed: number[][], nyquist: number, previousCorrected: number[][], options?: JH01Options): DealiasResult {
  const ref = shift2D(previousCorrected, options?.shiftAz ?? 0, options?.shiftRange ?? 0);
  const result = dealiasSweepZW06Js(observed, nyquist, ref);
  return { ...result, metadata: { ...result.metadata, method: "jh01", anchor: "previous_volume" } };
}

function dealiasSweepRegionGraphJs(observed: number[][], nyquist: number, options?: RegionGraphOptions): DealiasResult {
  const reference = options?.reference ? (isPackedSweep(options.reference) ? unpackSweep(options.reference) : options.reference) : undefined;
  const result = dealiasSweepZW06Js(observed, nyquist, reference);
  return { ...result, metadata: { ...result.metadata, method: "region_graph", approximation: "zw06_proxy" } };
}

function dealiasSweepRecursiveJs(observed: number[][], nyquist: number, options?: RecursiveOptions): DealiasResult {
  const result = dealiasSweepRegionGraphJs(observed, nyquist, options);
  return { ...result, metadata: { ...result.metadata, method: "recursive", approximation: "region_graph_proxy" } };
}

function dealiasSweepVariationalJs(observed: number[][], nyquist: number, options?: VariationalOptions): DealiasResult {
  const result = dealiasSweepRegionGraphJs(observed, nyquist, options);
  return { ...result, metadata: { ...result.metadata, method: "variational", approximation: "region_graph_proxy" } };
}

function dealiasSweepMLJs(observed: number[][], nyquist: number, options?: MlOptions): DealiasResult {
  const result = dealiasSweepVariationalJs(observed, nyquist, options);
  return { ...result, metadata: { ...result.metadata, method: "ml", approximation: "variational_proxy" } };
}

function dealiasDualPrfJs(lowObserved: number[][], highObserved: number[][], lowNyquist: number, highNyquist: number, options?: DualPrfOptions): DealiasResult {
  ensurePositiveNyquist(lowNyquist);
  ensurePositiveNyquist(highNyquist);
  const reference = options?.reference ? (isPackedSweep(options.reference) ? unpackSweep(options.reference) : options.reference) : undefined;
  const nAz = lowObserved.length;
  const nR = lowObserved[0].length;
  const corrected = Array.from({ length: nAz }, () => new Array<number>(nR).fill(Number.NaN));
  const folds = Array.from({ length: nAz }, () => new Array<number>(nR).fill(0));
  const confidence = Array.from({ length: nAz }, () => new Array<number>(nR).fill(0));
  const maxAbsFold = options?.maxAbsFold ?? 8;
  for (let i = 0; i < nAz; i++) {
    for (let j = 0; j < nR; j++) {
      const low = lowObserved[i][j];
      const high = highObserved[i][j];
      if (!isFiniteNumber(low) || !isFiniteNumber(high)) continue;
      let best = high;
      let bestScore = Number.POSITIVE_INFINITY;
      let bestFold = 0;
      for (let k = -maxAbsFold; k <= maxAbsFold; k++) {
        const candidate = high + 2 * highNyquist * k;
        const lowMismatch = Math.abs(wrapToNyquist(candidate, lowNyquist) - low);
        const refMismatch = reference && isFiniteNumber(reference[i][j]) ? Math.abs(candidate - reference[i][j]) : 0;
        const score = lowMismatch + 0.25 * refMismatch;
        if (score < bestScore) {
          bestScore = score;
          best = candidate;
          bestFold = k;
        }
      }
      corrected[i][j] = best;
      folds[i][j] = bestFold;
      confidence[i][j] = Math.exp(-0.5 * (bestScore / Math.max(lowNyquist, 1e-6)) ** 2);
    }
  }
  return makeMatrixResult(corrected, folds, confidence, { method: "dual_prf" });
}

function perSweepNyquist(nyquist: number | ArrayLike<number>, sweepIndex: number): number {
  if (typeof nyquist === "number") return nyquist;
  const value = nyquist[sweepIndex];
  if (typeof value !== "number") {
    throw new Error(`missing nyquist for sweep ${sweepIndex}`);
  }
  return value;
}

function dealiasVolumeJH01Js(observed: number[][][], nyquist: number | ArrayLike<number>, previousCorrected: number[][][], _options?: VolumeOptions): VolumeDealiasResult {
  const velocity: number[][][] = [];
  const folds: number[][][] = [];
  const confidence: number[][][] = [];
  for (let sweep = 0; sweep < observed.length; sweep++) {
    const result = dealiasSweepJH01Js(observed[sweep], perSweepNyquist(nyquist, sweep), previousCorrected[sweep]);
    velocity.push(result.velocity);
    folds.push(result.folds);
    confidence.push(result.confidence);
  }
  return { velocity, folds, confidence, metadata: { method: "jh01_volume", approximation: "per_sweep_loop" } };
}

function dealiasVolume3DJs(observed: number[][][], nyquist: number | ArrayLike<number>, options?: VolumeOptions): VolumeDealiasResult {
  const referenceVolume = options?.referenceVolume
    ? (isPackedVolume(options.referenceVolume) ? unpackVolume(options.referenceVolume) : options.referenceVolume)
    : undefined;
  const velocity: number[][][] = [];
  const folds: number[][][] = [];
  const confidence: number[][][] = [];
  for (let sweep = 0; sweep < observed.length; sweep++) {
    const reference = referenceVolume?.[sweep];
    const result = dealiasSweepZW06Js(observed[sweep], perSweepNyquist(nyquist, sweep), reference);
    velocity.push(result.velocity);
    folds.push(result.folds);
    confidence.push(result.confidence);
  }
  return { velocity, folds, confidence, metadata: { method: "volume_3d", approximation: "per_sweep_zw06" } };
}

function jsPacked(method: string, result: DealiasResult): PackedDealiasResult {
  return {
    ...packedResultFromMatrix(result),
    metadata: { backend: "js", method, ...result.metadata },
  };
}

function jsPackedVelocity(method: string, result: DealiasResult): PackedVelocityResult {
  return {
    velocity: Float32Array.from(packSweep(result.velocity).data),
    azimuthCount: result.velocity.length,
    gateCount: result.velocity[0]?.length ?? 0,
    metadata: { backend: "js", method, output: "velocity_only", ...result.metadata },
  };
}

function jsPackedVolume(method: string, result: VolumeDealiasResult): PackedVolumeDealiasResult {
  return {
    ...packedVolumeResultFromMatrix(result),
    metadata: { backend: "js", method, ...result.metadata },
  };
}

export function createJsBackend(): OpenDealiasBackend {
  return {
    name: "open-dealias-js",
    kind: "js",
    wrapToNyquist,
    unfoldToReference,
    foldCount,
    dealiasRadialES90: (observed, nyquist, reference) => dealiasRadialES90Js(observed, nyquist, reference),
    estimateUniformWindVAD: (observed, azimuthDeg, elevationDeg) => estimateUniformWindVAD(observed, azimuthDeg, elevationDeg),
    dealiasSweepZW06Packed: (observed, nyquist, reference) => jsPacked("zw06", dealiasSweepZW06Js(unpackSweep(observed), nyquist, reference ? unpackSweep(reference) : undefined)),
    dealiasSweepZW06VelocityPacked: (observed, nyquist, reference) => jsPackedVelocity("zw06", dealiasSweepZW06Js(unpackSweep(observed), nyquist, reference ? unpackSweep(reference) : undefined)),
    dealiasSweepXu11Packed: (observed, nyquist, options) => jsPacked("xu11", dealiasSweepXu11Js(unpackSweep(observed), nyquist, options)),
    dealiasSweepXu11VelocityPacked: (observed, nyquist, options) => jsPackedVelocity("xu11", dealiasSweepXu11Js(unpackSweep(observed), nyquist, options)),
    dealiasSweepJH01Packed: (observed, nyquist, previousCorrected, options) => jsPacked("jh01", dealiasSweepJH01Js(unpackSweep(observed), nyquist, unpackSweep(previousCorrected), options)),
    dealiasSweepJH01VelocityPacked: (observed, nyquist, previousCorrected, options) => jsPackedVelocity("jh01", dealiasSweepJH01Js(unpackSweep(observed), nyquist, unpackSweep(previousCorrected), options)),
    dealiasSweepRegionGraphPacked: (observed, nyquist, options) => jsPacked("region_graph", dealiasSweepRegionGraphJs(unpackSweep(observed), nyquist, options)),
    dealiasSweepRegionGraphVelocityPacked: (observed, nyquist, options) => jsPackedVelocity("region_graph", dealiasSweepRegionGraphJs(unpackSweep(observed), nyquist, options)),
    dealiasSweepRecursivePacked: (observed, nyquist, options) => jsPacked("recursive", dealiasSweepRecursiveJs(unpackSweep(observed), nyquist, options)),
    dealiasSweepVariationalPacked: (observed, nyquist, options) => jsPacked("variational", dealiasSweepVariationalJs(unpackSweep(observed), nyquist, options)),
    dealiasSweepVariationalVelocityPacked: (observed, nyquist, options) => jsPackedVelocity("variational", dealiasSweepVariationalJs(unpackSweep(observed), nyquist, options)),
    dealiasSweepMLPacked: (observed, nyquist, options) => jsPacked("ml", dealiasSweepMLJs(unpackSweep(observed), nyquist, options)),
    dealiasDualPrfPacked: (lowObserved, highObserved, lowNyquist, highNyquist, options) => jsPacked("dual_prf", dealiasDualPrfJs(unpackSweep(lowObserved), unpackSweep(highObserved), lowNyquist, highNyquist, options)),
    dealiasVolumeJH01Packed: (observed, nyquist, previousCorrected, options) => jsPackedVolume("jh01_volume", dealiasVolumeJH01Js(unpackVolume(observed), nyquist, unpackVolume(previousCorrected), options)),
    dealiasVolume3DPacked: (observed, nyquist, options) => jsPackedVolume("volume_3d", dealiasVolume3DJs(unpackVolume(observed), nyquist, options)),
  };
}

function getBackendMethod<T extends keyof OpenDealiasBackend>(backend: OpenDealiasBackend, method: T): OpenDealiasBackend[T] | undefined {
  return backend[method];
}

function lookupAlias(module: Record<string, unknown>, names: readonly string[]): ((...args: unknown[]) => unknown) | undefined {
  for (const name of names) {
    const candidate = module[name];
    if (typeof candidate === "function") return candidate as (...args: unknown[]) => unknown;
  }
  return undefined;
}

function isBackendObject(value: unknown): value is OpenDealiasBackend {
  return typeof value === "object" && value !== null && typeof (value as { name?: unknown }).name === "string";
}

function moduleBackendFromExports(module: Record<string, unknown>): OpenDealiasBackend {
  const name = typeof module.name === "string" ? module.name : "open-dealias-wasm";
  return {
    name,
    kind: "wasm",
    wrapToNyquist: lookupAlias(module, WASM_METHOD_ALIASES.wrapToNyquist) as OpenDealiasBackend["wrapToNyquist"],
    unfoldToReference: lookupAlias(module, WASM_METHOD_ALIASES.unfoldToReference) as OpenDealiasBackend["unfoldToReference"],
    foldCount: lookupAlias(module, WASM_METHOD_ALIASES.foldCount) as OpenDealiasBackend["foldCount"],
    estimateUniformWindVAD: lookupAlias(module, WASM_METHOD_ALIASES.estimateUniformWindVAD) as OpenDealiasBackend["estimateUniformWindVAD"],
    dealiasSweepZW06Packed: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepZW06Packed) as OpenDealiasBackend["dealiasSweepZW06Packed"],
    dealiasSweepZW06VelocityPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepZW06VelocityPacked) as OpenDealiasBackend["dealiasSweepZW06VelocityPacked"],
    dealiasSweepXu11Packed: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepXu11Packed) as OpenDealiasBackend["dealiasSweepXu11Packed"],
    dealiasSweepXu11VelocityPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepXu11VelocityPacked) as OpenDealiasBackend["dealiasSweepXu11VelocityPacked"],
    dealiasSweepJH01Packed: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepJH01Packed) as OpenDealiasBackend["dealiasSweepJH01Packed"],
    dealiasSweepJH01VelocityPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepJH01VelocityPacked) as OpenDealiasBackend["dealiasSweepJH01VelocityPacked"],
    dealiasSweepRegionGraphPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepRegionGraphPacked) as OpenDealiasBackend["dealiasSweepRegionGraphPacked"],
    dealiasSweepRegionGraphVelocityPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepRegionGraphVelocityPacked) as OpenDealiasBackend["dealiasSweepRegionGraphVelocityPacked"],
    dealiasSweepRecursivePacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepRecursivePacked) as OpenDealiasBackend["dealiasSweepRecursivePacked"],
    dealiasSweepVariationalPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepVariationalPacked) as OpenDealiasBackend["dealiasSweepVariationalPacked"],
    dealiasSweepVariationalVelocityPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepVariationalVelocityPacked) as OpenDealiasBackend["dealiasSweepVariationalVelocityPacked"],
    dealiasSweepMLPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasSweepMLPacked) as OpenDealiasBackend["dealiasSweepMLPacked"],
    dealiasDualPrfPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasDualPrfPacked) as OpenDealiasBackend["dealiasDualPrfPacked"],
    dealiasVolumeJH01Packed: lookupAlias(module, WASM_METHOD_ALIASES.dealiasVolumeJH01Packed) as OpenDealiasBackend["dealiasVolumeJH01Packed"],
    dealiasVolume3DPacked: lookupAlias(module, WASM_METHOD_ALIASES.dealiasVolume3DPacked) as OpenDealiasBackend["dealiasVolume3DPacked"],
  };
}

async function resolveBackendSource(source: OpenDealiasBackendSource): Promise<OpenDealiasBackend> {
  const resolved = typeof source === "function"
    ? await source()
    : await source;

  if (isBackendObject(resolved)) {
    return resolved;
  }

  if (typeof resolved !== "object" || resolved === null) {
    throw new Error("backend source did not resolve to an object");
  }

  const module = resolved as Record<string, unknown>;
  if (typeof module.default === "function") {
    await (module.default as (input?: unknown) => Promise<unknown> | unknown)();
  }

  if (typeof module.createOpenDealiasBackend === "function") {
    const backend = await (module.createOpenDealiasBackend as () => Promise<OpenDealiasBackend> | OpenDealiasBackend)();
    if (!isBackendObject(backend)) {
      throw new Error("createOpenDealiasBackend did not return a valid backend");
    }
    return backend;
  }

  return moduleBackendFromExports(module);
}

export function getOpenDealiasBackend(): OpenDealiasBackend {
  if (activeBackend === null) {
    activeBackend = createJsBackend();
  }
  return activeBackend;
}

export function setOpenDealiasBackend(backend: OpenDealiasBackend): void {
  activeBackend = backend;
}

export function resetOpenDealiasBackend(): void {
  activeBackend = createJsBackend();
}

export function hasCustomOpenDealiasBackend(): boolean {
  return getOpenDealiasBackend().kind !== "js";
}

export async function initOpenDealiasWasm(source: OpenDealiasBackendSource): Promise<OpenDealiasBackend> {
  const backend = await resolveBackendSource(source);
  setOpenDealiasBackend(backend);
  return backend;
}

export function dealiasRadialES90(
  observed: ArrayLike<number>,
  nyquist: number,
  reference?: ArrayLike<number>,
): { velocity: number[]; folds: number[]; confidence: number[]; metadata: Record<string, unknown> } {
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasRadialES90");
  if (typeof method === "function") {
    const result = method.call(backend, observed, nyquist, reference);
    return {
      velocity: Array.from(result.velocity),
      folds: Array.from(result.folds),
      confidence: Array.from(result.confidence),
      metadata: { backend: backend.name, ...(result.metadata ?? {}) },
    };
  }
  return dealiasRadialES90Js(observed, nyquist, reference);
}

export function dealiasSweepZW06Packed(observed: PackedSweep | number[][], nyquist: number, reference?: PackedSweep | number[][]): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const packedReference = packReference(reference);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepZW06Packed");
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, packedReference);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepZW06Packed!(packedObserved, nyquist, packedReference);
}

export function dealiasSweepZW06VelocityPacked(observed: PackedSweep | number[][], nyquist: number, reference?: PackedSweep | number[][]): PackedVelocityResult {
  const packedObserved = packSweep(observed);
  const packedReference = packReference(reference);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepZW06VelocityPacked");
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, packedReference);
    return normalizePackedVelocityResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepZW06VelocityPacked!(packedObserved, nyquist, packedReference);
}

export function dealiasSweepZW06(observed: number[][], nyquist: number, reference?: number[][]): DealiasResult {
  return unpackPackedResult(dealiasSweepZW06Packed(observed, nyquist, reference));
}

export function dealiasSweepXu11Packed(observed: PackedSweep | number[][], nyquist: number, options: Xu11Options): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepXu11Packed");
  const normalizedOptions: Xu11Options = {
    ...options,
    reference: packReference(options.reference),
  };
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepXu11Packed!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepXu11VelocityPacked(observed: PackedSweep | number[][], nyquist: number, options: Xu11Options): PackedVelocityResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepXu11VelocityPacked");
  const normalizedOptions: Xu11Options = {
    ...options,
    reference: packReference(options.reference),
  };
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedVelocityResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepXu11VelocityPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepXu11(observed: number[][], nyquist: number, options: Xu11Options): DealiasResult {
  return unpackPackedResult(dealiasSweepXu11Packed(observed, nyquist, options));
}

export function dealiasSweepJH01Packed(observed: PackedSweep | number[][], nyquist: number, previousCorrected: PackedSweep | number[][], options?: JH01Options): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const packedPrevious = packSweep(previousCorrected);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepJH01Packed");
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, packedPrevious, options);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepJH01Packed!(packedObserved, nyquist, packedPrevious, options);
}

export function dealiasSweepJH01VelocityPacked(observed: PackedSweep | number[][], nyquist: number, previousCorrected: PackedSweep | number[][], options?: JH01Options): PackedVelocityResult {
  const packedObserved = packSweep(observed);
  const packedPrevious = packSweep(previousCorrected);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepJH01VelocityPacked");
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, packedPrevious, options);
    return normalizePackedVelocityResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepJH01VelocityPacked!(packedObserved, nyquist, packedPrevious, options);
}

export function dealiasSweepJH01(observed: number[][], nyquist: number, previousCorrected: number[][], options?: JH01Options): DealiasResult {
  return unpackPackedResult(dealiasSweepJH01Packed(observed, nyquist, previousCorrected, options));
}

export function dealiasSweepRegionGraphPacked(observed: PackedSweep | number[][], nyquist: number, options?: RegionGraphOptions): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepRegionGraphPacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepRegionGraphPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepRegionGraphVelocityPacked(observed: PackedSweep | number[][], nyquist: number, options?: RegionGraphOptions): PackedVelocityResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepRegionGraphVelocityPacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedVelocityResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepRegionGraphVelocityPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepRegionGraph(observed: number[][], nyquist: number, options?: RegionGraphOptions): DealiasResult {
  return unpackPackedResult(dealiasSweepRegionGraphPacked(observed, nyquist, options));
}

export function dealiasSweepRecursivePacked(observed: PackedSweep | number[][], nyquist: number, options?: RecursiveOptions): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepRecursivePacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepRecursivePacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepRecursive(observed: number[][], nyquist: number, options?: RecursiveOptions): DealiasResult {
  return unpackPackedResult(dealiasSweepRecursivePacked(observed, nyquist, options));
}

export function dealiasSweepVariationalPacked(observed: PackedSweep | number[][], nyquist: number, options?: VariationalOptions): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepVariationalPacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepVariationalPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepVariationalVelocityPacked(observed: PackedSweep | number[][], nyquist: number, options?: VariationalOptions): PackedVelocityResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepVariationalVelocityPacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedVelocityResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepVariationalVelocityPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepVariational(observed: number[][], nyquist: number, options?: VariationalOptions): DealiasResult {
  return unpackPackedResult(dealiasSweepVariationalPacked(observed, nyquist, options));
}

export function dealiasSweepMLPacked(observed: PackedSweep | number[][], nyquist: number, options?: MlOptions): PackedDealiasResult {
  const packedObserved = packSweep(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasSweepMLPacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedResult(result, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasSweepMLPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasSweepML(observed: number[][], nyquist: number, options?: MlOptions): DealiasResult {
  return unpackPackedResult(dealiasSweepMLPacked(observed, nyquist, options));
}

export function dealiasDualPrfPacked(
  lowObserved: PackedSweep | number[][],
  highObserved: PackedSweep | number[][],
  lowNyquist: number,
  highNyquist: number,
  options?: DualPrfOptions,
): PackedDealiasResult {
  const packedLow = packSweep(lowObserved);
  const packedHigh = packSweep(highObserved);
  if (packedLow.azimuthCount !== packedHigh.azimuthCount || packedLow.gateCount !== packedHigh.gateCount) {
    throw new Error("dual-prf inputs must have matching shapes");
  }
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasDualPrfPacked");
  const normalizedOptions = options ? { ...options, reference: packReference(options.reference) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedLow, packedHigh, lowNyquist, highNyquist, normalizedOptions);
    return normalizePackedResult(result, packedLow.azimuthCount, packedLow.gateCount);
  }
  return createJsBackend().dealiasDualPrfPacked!(packedLow, packedHigh, lowNyquist, highNyquist, normalizedOptions);
}

export function dealiasDualPrf(
  lowObserved: number[][],
  highObserved: number[][],
  lowNyquist: number,
  highNyquist: number,
  options?: DualPrfOptions,
): DealiasResult {
  return unpackPackedResult(dealiasDualPrfPacked(lowObserved, highObserved, lowNyquist, highNyquist, options));
}

export function dealiasVolumeJH01Packed(
  observed: PackedVolume | number[][][],
  nyquist: number | ArrayLike<number>,
  previousCorrected: PackedVolume | number[][][],
  options?: VolumeOptions,
): PackedVolumeDealiasResult {
  const packedObserved = packVolume(observed);
  const packedPrevious = packVolume(previousCorrected);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasVolumeJH01Packed");
  const normalizedOptions = options ? { ...options, referenceVolume: packVolumeReference(options.referenceVolume), previousVolume: packVolumeReference(options.previousVolume) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, packedPrevious, normalizedOptions);
    return normalizePackedVolumeResult(result, packedObserved.sweepCount, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasVolumeJH01Packed!(packedObserved, nyquist, packedPrevious, normalizedOptions);
}

export function dealiasVolumeJH01(
  observed: number[][][],
  nyquist: number | ArrayLike<number>,
  previousCorrected: number[][][],
  options?: VolumeOptions,
): VolumeDealiasResult {
  return unpackPackedVolumeResult(dealiasVolumeJH01Packed(observed, nyquist, previousCorrected, options));
}

export function dealiasVolume3DPacked(
  observed: PackedVolume | number[][][],
  nyquist: number | ArrayLike<number>,
  options?: VolumeOptions,
): PackedVolumeDealiasResult {
  const packedObserved = packVolume(observed);
  const backend = getOpenDealiasBackend();
  const method = getBackendMethod(backend, "dealiasVolume3DPacked");
  const normalizedOptions = options ? { ...options, referenceVolume: packVolumeReference(options.referenceVolume), previousVolume: packVolumeReference(options.previousVolume) } : undefined;
  if (typeof method === "function") {
    const result = method.call(backend, packedObserved, nyquist, normalizedOptions);
    return normalizePackedVolumeResult(result, packedObserved.sweepCount, packedObserved.azimuthCount, packedObserved.gateCount);
  }
  return createJsBackend().dealiasVolume3DPacked!(packedObserved, nyquist, normalizedOptions);
}

export function dealiasVolume3D(
  observed: number[][][],
  nyquist: number | ArrayLike<number>,
  options?: VolumeOptions,
): VolumeDealiasResult {
  return unpackPackedVolumeResult(dealiasVolume3DPacked(observed, nyquist, options));
}
