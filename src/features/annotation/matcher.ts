import { MATCH_MARGIN, MATCH_THRESHOLD } from './model-config';
import type { Item } from './types';
import { dot } from './vector';

export type MatchResult = {
  /** Best item if it cleared both the threshold and the margin, otherwise null. */
  item: Item | null;
  /** Best raw score, shown on screen to help tune the thresholds. */
  score: number;
  /** Best candidate even when it was rejected. */
  candidate: Item | null;
};

/**
 * Nearest-neighbour match of a query embedding against saved items.
 *
 * Each item scores as the best similarity over its samples (one per captured
 * angle). The winner must clear `MATCH_THRESHOLD` and beat the runner-up item by
 * `MATCH_MARGIN`, so two similar-looking items don't flicker between labels.
 */
export function matchItem(query: Float32Array, items: Item[]): MatchResult {
  let best: Item | null = null;
  let bestScore = -Infinity;
  let secondScore = -Infinity;

  for (const item of items) {
    let itemScore = -Infinity;
    for (const sample of item.samples) {
      itemScore = Math.max(itemScore, dot(query, sample));
    }
    if (itemScore > bestScore) {
      secondScore = bestScore;
      bestScore = itemScore;
      best = item;
    } else if (itemScore > secondScore) {
      secondScore = itemScore;
    }
  }

  if (!best) return { item: null, score: 0, candidate: null };

  const accepted = bestScore >= MATCH_THRESHOLD && bestScore - secondScore >= MATCH_MARGIN;
  return { item: accepted ? best : null, score: bestScore, candidate: best };
}
