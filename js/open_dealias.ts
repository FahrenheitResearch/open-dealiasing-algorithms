export interface DealiasResult {
  velocity: number[][];
  folds: number[][];
  confidence: number[][];
  metadata: Record<string, unknown>;
}

function isFiniteNumber(x: number): boolean {
  return Number.isFinite(x);
}

export function wrapToNyquist(value: number, nyquist: number): number {
  if (!(nyquist > 0)) throw new Error("nyquist must be positive");
  if (!isFiniteNumber(value)) return Number.NaN;
  return ((value + nyquist) % (2 * nyquist) + 2 * nyquist) % (2 * nyquist) - nyquist;
}

export function unfoldToReference(observed: number, reference: number, nyquist: number): number {
  if (!(nyquist > 0)) throw new Error("nyquist must be positive");
  if (!isFiniteNumber(observed) || !isFiniteNumber(reference)) return Number.NaN;
  const fold = Math.round((reference - observed) / (2 * nyquist));
  return observed + 2 * nyquist * fold;
}

export function foldCount(unfolded: number, observed: number, nyquist: number): number {
  if (!isFiniteNumber(unfolded) || !isFiniteNumber(observed)) return 0;
  return Math.round((unfolded - observed) / (2 * nyquist));
}

function medianFinite(values: number[]): number {
  const finite = values.filter(isFiniteNumber).sort((a, b) => a - b);
  if (finite.length === 0) return Number.NaN;
  const mid = Math.floor(finite.length / 2);
  return finite.length % 2 ? finite[mid] : 0.5 * (finite[mid - 1] + finite[mid]);
}

export function dealiasRadialES90(
  observed: number[],
  nyquist: number,
  reference?: number[],
): { velocity: number[]; folds: number[]; confidence: number[] } {
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
    return { velocity: corrected, folds: new Array<number>(n).fill(0), confidence };
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
    folds: corrected.map((v, i) => foldCount(v, observed[i], nyquist)),
    confidence,
  };
}

function clone2D(field: number[][]): number[][] {
  return field.map((row) => row.slice());
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
  const vals: number[] = [];
  for (let di = -1; di <= 1; di++) {
    for (let dj = -1; dj <= 1; dj++) {
      if (di === 0 && dj === 0) continue;
      const ii = (i + di + nAz) % nAz;
      const jj = j + dj;
      if (jj < 0 || jj >= nR) continue;
      const v = field[ii][jj];
      if (isFiniteNumber(v)) vals.push(v);
    }
  }
  return vals;
}

export function dealiasSweepZW06(observed: number[][], nyquist: number, reference?: number[][]): DealiasResult {
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
        const cand = unfoldToReference(obs, reference[i][j], nyquist);
        if (Math.abs(cand - reference[i][j]) <= 0.65 * nyquist) {
          corrected[i][j] = cand;
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
          const cand = unfoldToReference(observed[i][j], ref, nyquist);
          const mismatch = Math.abs(cand - ref);
          if (mismatch <= pass.maxMismatch) {
            corrected[i][j] = cand;
            confidence[i][j] = Math.exp(-0.5 * (mismatch / (0.4 * nyquist)) ** 2);
            changed = true;
          }
        }
      }
    }
  }

  const folds = Array.from({ length: nAz }, () => new Array<number>(nR).fill(0));
  for (let i = 0; i < nAz; i++) {
    for (let j = 0; j < nR; j++) {
      folds[i][j] = foldCount(corrected[i][j], observed[i][j], nyquist);
    }
  }
  return { velocity: corrected, folds, confidence, metadata: { method: "2d_multipass" } };
}

export function buildReferenceFromUV(azimuthDeg: number[], nRange: number, u: number, v: number, elevationDeg = 0): number[][] {
  const el = elevationDeg * Math.PI / 180;
  return azimuthDeg.map((azDeg) => {
    const az = azDeg * Math.PI / 180;
    const vr = Math.cos(el) * (u * Math.sin(az) + v * Math.cos(az));
    return new Array<number>(nRange).fill(vr);
  });
}

export function dealiasSweepJH01(
  observed: number[][],
  nyquist: number,
  previousCorrected: number[][],
  shiftAz = 0,
  shiftRange = 0,
): DealiasResult {
  const ref = shift2D(previousCorrected, shiftAz, shiftRange);
  return dealiasSweepZW06(observed, nyquist, ref);
}
