/** A labeled real-world object, e.g. "Alvin's TV". */
export type Item = {
  id: number;
  label: string;
  createdAt: number;
  /** Normalized embeddings from the current model, one per captured angle. */
  samples: Float32Array[];
};

/** One captured view of an item: its embedding plus the crop it came from. */
export type CapturedSample = {
  vector: Float32Array;
  /** RGB, HWC, CROP_SIZE x CROP_SIZE. Kept so samples can be re-embedded by a new model. */
  pixels: Uint8Array;
};
