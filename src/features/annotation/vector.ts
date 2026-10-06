/** Dot product. Equals cosine similarity when both vectors are L2-normalized. */
export function dot(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

export function normalize(v: Float32Array): Float32Array {
  const norm = Math.sqrt(dot(v, v));
  if (norm === 0) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) {
    out[i] = v[i] / norm;
  }
  return out;
}

/** Stores a vector as raw float32 bytes for a SQLite BLOB column. */
export function vectorToBlob(v: Float32Array): Uint8Array {
  return new Uint8Array(v.buffer.slice(v.byteOffset, v.byteOffset + v.byteLength));
}

export function blobToVector(blob: Uint8Array): Float32Array {
  // Copy into a fresh, aligned buffer: the BLOB's byteOffset may not be a multiple of 4.
  return new Float32Array(blob.slice().buffer);
}
