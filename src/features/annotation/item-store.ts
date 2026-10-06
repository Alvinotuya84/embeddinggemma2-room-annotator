import type { SQLiteDatabase } from 'expo-sqlite';

import type { CapturedSample, Item } from './types';
import { blobToVector, vectorToBlob } from './vector';

export const DATABASE_NAME = 'annotations.db';

const DATABASE_VERSION = 1;

export async function migrateDbIfNeeded(db: SQLiteDatabase) {
  // Foreign keys are off by default in SQLite and must be enabled per connection.
  await db.execAsync('PRAGMA foreign_keys = ON');

  const result = await db.getFirstAsync<{ user_version: number }>('PRAGMA user_version');
  const currentDbVersion = result?.user_version ?? 0;
  if (currentDbVersion >= DATABASE_VERSION) {
    return;
  }
  if (currentDbVersion === 0) {
    await db.execAsync(`
PRAGMA journal_mode = 'wal';
CREATE TABLE items (
  id INTEGER PRIMARY KEY NOT NULL,
  label TEXT NOT NULL,
  created_at INTEGER NOT NULL
);
CREATE TABLE samples (
  id INTEGER PRIMARY KEY NOT NULL,
  item_id INTEGER NOT NULL REFERENCES items(id) ON DELETE CASCADE,
  model TEXT NOT NULL,
  vector BLOB NOT NULL,
  pixels BLOB NOT NULL,
  width INTEGER NOT NULL,
  height INTEGER NOT NULL
);
CREATE INDEX samples_item_model ON samples(item_id, model);
`);
  }
  await db.execAsync(`PRAGMA user_version = ${DATABASE_VERSION}`);
}

/** Loads all items with the sample vectors produced by `modelId`. Pixels are not loaded. */
export async function loadItems(db: SQLiteDatabase, modelId: string): Promise<Item[]> {
  const itemRows = await db.getAllAsync<{ id: number; label: string; created_at: number }>(
    'SELECT id, label, created_at FROM items ORDER BY created_at DESC'
  );
  const sampleRows = await db.getAllAsync<{ item_id: number; vector: Uint8Array }>(
    'SELECT item_id, vector FROM samples WHERE model = ?',
    modelId
  );

  const samplesByItem = new Map<number, Float32Array[]>();
  for (const row of sampleRows) {
    const list = samplesByItem.get(row.item_id) ?? [];
    list.push(blobToVector(row.vector));
    samplesByItem.set(row.item_id, list);
  }

  return itemRows.map((row) => ({
    id: row.id,
    label: row.label,
    createdAt: row.created_at,
    samples: samplesByItem.get(row.id) ?? [],
  }));
}

export async function insertItem(
  db: SQLiteDatabase,
  modelId: string,
  label: string,
  samples: CapturedSample[],
  cropSize: number
): Promise<void> {
  await db.withExclusiveTransactionAsync(async (txn) => {
    const { lastInsertRowId } = await txn.runAsync(
      'INSERT INTO items (label, created_at) VALUES (?, ?)',
      label,
      Date.now()
    );
    for (const sample of samples) {
      await txn.runAsync(
        'INSERT INTO samples (item_id, model, vector, pixels, width, height) VALUES (?, ?, ?, ?, ?, ?)',
        lastInsertRowId,
        modelId,
        vectorToBlob(sample.vector),
        sample.pixels,
        cropSize,
        cropSize
      );
    }
  });
}

export async function deleteItem(db: SQLiteDatabase, id: number): Promise<void> {
  await db.runAsync('DELETE FROM items WHERE id = ?', id);
}

