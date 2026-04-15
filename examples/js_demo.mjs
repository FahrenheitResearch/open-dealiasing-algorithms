import { buildReferenceFromUV, dealiasSweepZW06, wrapToNyquist } from "../js/dist/open_dealias.js";

const azimuth = Array.from({ length: 180 }, (_, i) => i * 2);
const truth = buildReferenceFromUV(azimuth, 60, 18, -3);
const observed = truth.map((row) => row.map((v) => wrapToNyquist(v, 10)));
const result = dealiasSweepZW06(observed, 10, truth);

let mae = 0;
let n = 0;
for (let i = 0; i < truth.length; i++) {
  for (let j = 0; j < truth[0].length; j++) {
    mae += Math.abs(result.velocity[i][j] - truth[i][j]);
    n += 1;
  }
}
mae /= n;
console.log({ mae, metadata: result.metadata });
