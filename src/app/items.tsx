import { Alert, FlatList, Pressable, StyleSheet, View } from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, MaxContentWidth, Spacing } from '@/constants/theme';
import { useItems } from '@/features/annotation/items-context';
import type { Item } from '@/features/annotation/types';

export default function ItemsScreen() {
  const insets = useSafeAreaInsets();
  const { items, isLoaded, removeItem } = useItems();

  const confirmDelete = (item: Item) => {
    Alert.alert(`Delete “${item.label}”?`, 'Its saved views will be removed from this device.', [
      { text: 'Cancel', style: 'cancel' },
      { text: 'Delete', style: 'destructive', onPress: () => removeItem(item.id) },
    ]);
  };

  return (
    <ThemedView style={styles.container}>
      <FlatList
        data={items}
        keyExtractor={(item) => String(item.id)}
        contentContainerStyle={[
          styles.content,
          {
            paddingTop: insets.top + Spacing.four,
            paddingBottom: insets.bottom + BottomTabInset + Spacing.three,
          },
        ]}
        ListHeaderComponent={<ThemedText type="subtitle">Saved items</ThemedText>}
        ListEmptyComponent={
          isLoaded ? (
            <ThemedText themeColor="textSecondary">
              No items yet. Go to Scan and tap “Add item”.
            </ThemedText>
          ) : null
        }
        renderItem={({ item }) => (
          <ThemedView type="backgroundElement" style={styles.row}>
            <View style={styles.rowText}>
              <ThemedText type="default">{item.label}</ThemedText>
              <ThemedText type="small" themeColor="textSecondary">
                {item.samples.length === 0
                  ? 'Needs re-capturing for the current model'
                  : `${item.samples.length} views · added ${new Date(item.createdAt).toLocaleDateString()}`}
              </ThemedText>
            </View>
            <Pressable
              accessibilityRole="button"
              accessibilityLabel={`Delete ${item.label}`}
              hitSlop={Spacing.two}
              onPress={() => confirmDelete(item)}>
              <ThemedText type="smallBold" style={styles.delete}>
                Delete
              </ThemedText>
            </Pressable>
          </ThemedView>
        )}
      />
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    gap: Spacing.two,
    paddingHorizontal: Spacing.three,
    width: '100%',
    maxWidth: MaxContentWidth,
    alignSelf: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.three,
    padding: Spacing.three,
    borderRadius: Spacing.three,
  },
  rowText: {
    flex: 1,
  },
  delete: {
    color: '#E5484D',
  },
});
