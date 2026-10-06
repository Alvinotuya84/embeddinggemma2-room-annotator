import type { ImageEmbedderModel } from 'react-native-executorch';

/**
 * The single place that decides which image embedding model the app uses.
 *
 * Bump `EMBEDDING_MODEL_ID` whenever the model changes: saved samples are
 * tagged with the id they were embedded with, and samples from another model
 * are ignored because their vectors live in a different space.
 *
 * To go back to CLIP, use `models.imageEmbeddings.CLIP_VIT_BASE_PATCH32.DEFAULT`
 * (from `react-native-executorch`) with the id `'clip-vit-b32'`.
 */
export const EMBEDDING_MODEL_ID = 'embeddinggemma2-384';

/**
 * EmbeddingGemma 2 image embedder exported by tools/export-embeddinggemma2.
 * The URL is pinned to a commit: downloads are cached by URL, so a re-export
 * must get a new URL or devices keep the cached file.
 */
const EMBEDDINGGEMMA2_COMMIT = '3c848d4902b4b38012af3fdd8d40d8deabae9a76';
export const IMAGE_EMBEDDER: ImageEmbedderModel = {
  modelPath: `https://huggingface.co/GeekCrick/embeddinggemma2-rne/resolve/${EMBEDDINGGEMMA2_COMMIT}/embeddinggemma2_image_384.pte`,
  modelOpts: {
    // Crops already arrive at the model's input size, so this resize is a no-op.
    resizeMode: 'stretch',
    interpolation: 'cubic',
    // RGB 0..255 -> 0..1. The model applies Gemma's own 2 * (x - 0.5) scaling internally.
    normalizeOpts: { alpha: 1 / 255, beta: 0 },
  },
};

/**
 * Size of the centered square crop taken from each camera frame. Must equal the
 * exported model's input size (`--side` in the export script).
 */
export const CROP_SIZE = 384;

/**
 * Minimum time between two live embeddings, to limit heat and battery drain.
 * EmbeddingGemma 2 is much heavier than CLIP; frames that arrive while a run is
 * still busy are dropped anyway.
 */
export const SCAN_INTERVAL_MS = 1000;

/** Number of angles to capture when saving a new item. */
export const MIN_SAMPLES = 3;
export const MAX_SAMPLES = 6;

/**
 * Match thresholds for image-to-image cosine similarity. Starting points only;
 * tune them on real objects (the scan screen shows the raw score).
 */
export const MATCH_THRESHOLD = 0.85;
export const MATCH_MARGIN = 0.03;
