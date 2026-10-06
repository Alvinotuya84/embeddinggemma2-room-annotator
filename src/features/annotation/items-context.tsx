import { SQLiteProvider, useSQLiteContext } from 'expo-sqlite';
import { createContext, use, useCallback, useEffect, useState, type ReactNode } from 'react';

import * as store from './item-store';
import { CROP_SIZE, EMBEDDING_MODEL_ID } from './model-config';
import type { CapturedSample, Item } from './types';

type ItemsContextValue = {
  items: Item[];
  isLoaded: boolean;
  addItem: (label: string, samples: CapturedSample[]) => Promise<void>;
  removeItem: (id: number) => Promise<void>;
};

const ItemsContext = createContext<ItemsContextValue | null>(null);

/** Owns the SQLite connection and keeps saved items in memory for fast matching. */
export function ItemsProvider({ children }: { children: ReactNode }) {
  return (
    <SQLiteProvider databaseName={store.DATABASE_NAME} onInit={store.migrateDbIfNeeded}>
      <ItemsStateProvider>{children}</ItemsStateProvider>
    </SQLiteProvider>
  );
}

function ItemsStateProvider({ children }: { children: ReactNode }) {
  const db = useSQLiteContext();
  const [items, setItems] = useState<Item[]>([]);
  const [isLoaded, setIsLoaded] = useState(false);

  const reload = useCallback(async () => {
    setItems(await store.loadItems(db, EMBEDDING_MODEL_ID));
    setIsLoaded(true);
  }, [db]);

  useEffect(() => {
    let cancelled = false;
    store.loadItems(db, EMBEDDING_MODEL_ID).then((loaded) => {
      if (cancelled) return;
      setItems(loaded);
      setIsLoaded(true);
    });
    return () => {
      cancelled = true;
    };
  }, [db]);

  const addItem = useCallback(
    async (label: string, samples: CapturedSample[]) => {
      await store.insertItem(db, EMBEDDING_MODEL_ID, label, samples, CROP_SIZE);
      await reload();
    },
    [db, reload]
  );

  const removeItem = useCallback(
    async (id: number) => {
      await store.deleteItem(db, id);
      await reload();
    },
    [db, reload]
  );

  return (
    <ItemsContext value={{ items, isLoaded, addItem, removeItem }}>
      {children}
    </ItemsContext>
  );
}

export function useItems(): ItemsContextValue {
  const value = use(ItemsContext);
  if (!value) throw new Error('useItems must be used inside <ItemsProvider>');
  return value;
}
