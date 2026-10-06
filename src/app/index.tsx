import { useIsFocused } from 'expo-router';
import { useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Linking,
  Platform,
  Pressable,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { useSafeAreaInsets } from 'react-native-safe-area-context';
import { Camera, useCameraDevice, useCameraPermission } from 'react-native-vision-camera';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { BottomTabInset, Spacing } from '@/constants/theme';
import { useItems } from '@/features/annotation/items-context';
import { matchItem, type MatchResult } from '@/features/annotation/matcher';
import { MAX_SAMPLES, MIN_SAMPLES } from '@/features/annotation/model-config';
import type { CapturedSample } from '@/features/annotation/types';
import { useFrameEmbedder } from '@/features/annotation/use-frame-embedder';
import { normalize } from '@/features/annotation/vector';
import { useAppIsActive } from '@/hooks/use-app-is-active';

type Mode = 'scan' | 'capture' | 'name';

const MATCH_COLOR = '#30D158';
const IDLE_COLOR = 'rgba(255,255,255,0.85)';

export default function ScanScreen() {
  const { hasPermission, canRequestPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice('back');
  const isFocused = useIsFocused();
  const isAppActive = useAppIsActive();
  const insets = useSafeAreaInsets();
  const { items, addItem } = useItems();

  const [mode, setMode] = useState<Mode>('scan');
  const [match, setMatch] = useState<MatchResult | null>(null);
  const [captured, setCaptured] = useState<CapturedSample[]>([]);
  const [label, setLabel] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  const embedder = useFrameEmbedder({
    onEmbedding: (vector) => {
      if (mode !== 'scan') return;
      setMatch(matchItem(normalize(vector), items));
    },
    onCapture: (sample) => {
      setCaptured((prev) =>
        prev.length >= MAX_SAMPLES ? prev : [...prev, { ...sample, vector: normalize(sample.vector) }]
      );
    },
  });

  const resetCapture = () => {
    setCaptured([]);
    setLabel('');
    setMode('scan');
  };

  const save = async () => {
    const trimmed = label.trim();
    if (!trimmed || captured.length < MIN_SAMPLES) return;
    setIsSaving(true);
    try {
      await addItem(trimmed, captured);
      resetCapture();
    } finally {
      setIsSaving(false);
    }
  };

  if (!hasPermission) {
    return (
      <ThemedView style={styles.centered}>
        <ThemedText type="subtitle">Camera access</ThemedText>
        <ThemedText themeColor="textSecondary" style={styles.centeredText}>
          The camera is used to recognize the items you save. Everything runs on this device.
        </ThemedText>
        <Pressable
          style={styles.primaryButton}
          onPress={() => (canRequestPermission ? requestPermission() : Linking.openSettings())}>
          <Text style={styles.primaryButtonText}>
            {canRequestPermission ? 'Allow camera' : 'Open Settings'}
          </Text>
        </Pressable>
      </ThemedView>
    );
  }

  if (!device) {
    return (
      <ThemedView style={styles.centered}>
        <ThemedText>No back camera found on this device.</ThemedText>
      </ThemedView>
    );
  }

  const matched = mode === 'scan' && match?.item != null;
  const reticleColor = matched ? MATCH_COLOR : IDLE_COLOR;

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={isFocused && isAppActive}
        orientationSource="interface"
        outputs={[embedder.frameOutput]}
        resizeMode="cover"
        enableNativeTapToFocusGesture
      />

      <View style={[styles.topBar, { paddingTop: insets.top + Spacing.two }]} pointerEvents="none">
        <StatusPill embedder={embedder} itemCount={items.length} />
      </View>

      <View style={styles.reticleArea} pointerEvents="none">
        <View style={[styles.reticle, { borderColor: reticleColor }]} />
        {mode === 'scan' && <MatchLabel match={match} itemCount={items.length} />}
      </View>

      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        style={[styles.bottomBar, { paddingBottom: BottomTabInset + insets.bottom + Spacing.three }]}>
        {mode === 'scan' && (
          <Pressable
            style={[styles.primaryButton, !embedder.isReady && styles.disabled]}
            disabled={!embedder.isReady}
            onPress={() => setMode('capture')}>
            <Text style={styles.primaryButtonText}>Add item</Text>
          </Pressable>
        )}

        {mode === 'capture' && (
          <View style={styles.panel}>
            <Text style={styles.panelText}>
              Fill the square with the item and capture it from {MIN_SAMPLES}–{MAX_SAMPLES} angles.
            </Text>
            <View style={styles.row}>
              <Pressable style={styles.secondaryButton} onPress={resetCapture}>
                <Text style={styles.secondaryButtonText}>Cancel</Text>
              </Pressable>
              <Pressable
                accessibilityLabel={`Capture angle ${captured.length + 1}`}
                style={[styles.shutter, captured.length >= MAX_SAMPLES && styles.disabled]}
                disabled={captured.length >= MAX_SAMPLES}
                onPress={embedder.requestCapture}>
                <Text style={styles.shutterText}>
                  {captured.length}/{MAX_SAMPLES}
                </Text>
              </Pressable>
              <Pressable
                style={[styles.secondaryButton, captured.length < MIN_SAMPLES && styles.disabled]}
                disabled={captured.length < MIN_SAMPLES}
                onPress={() => setMode('name')}>
                <Text style={styles.secondaryButtonText}>Next</Text>
              </Pressable>
            </View>
          </View>
        )}

        {mode === 'name' && (
          <View style={styles.panel}>
            <Text style={styles.panelText}>What is this item?</Text>
            <TextInput
              autoFocus
              value={label}
              onChangeText={setLabel}
              placeholder="e.g. Alvin's TV"
              placeholderTextColor="rgba(255,255,255,0.5)"
              style={styles.input}
              returnKeyType="done"
              onSubmitEditing={save}
            />
            <View style={styles.row}>
              <Pressable style={styles.secondaryButton} onPress={() => setMode('capture')}>
                <Text style={styles.secondaryButtonText}>Back</Text>
              </Pressable>
              <Pressable
                style={[styles.primaryButton, (!label.trim() || isSaving) && styles.disabled]}
                disabled={!label.trim() || isSaving}
                onPress={save}>
                <Text style={styles.primaryButtonText}>{isSaving ? 'Saving…' : 'Save'}</Text>
              </Pressable>
            </View>
          </View>
        )}
      </KeyboardAvoidingView>
    </View>
  );
}

function StatusPill({
  embedder,
  itemCount,
}: {
  embedder: ReturnType<typeof useFrameEmbedder>;
  itemCount: number;
}) {
  let text: string;
  if (embedder.error) {
    text = `Model error: ${embedder.error.message}`;
  } else if (!embedder.isReady) {
    text =
      embedder.downloadProgress < 100
        ? `Downloading model ${Math.round(embedder.downloadProgress)}%`
        : 'Loading model…';
  } else {
    text = itemCount === 1 ? '1 item saved' : `${itemCount} items saved`;
  }
  return (
    <View style={styles.pill}>
      {!embedder.isReady && !embedder.error && <ActivityIndicator color="#fff" size="small" />}
      <Text style={styles.pillText}>{text}</Text>
    </View>
  );
}

function MatchLabel({ match, itemCount }: { match: MatchResult | null; itemCount: number }) {
  if (itemCount === 0) {
    return <Text style={styles.hint}>Tap “Add item” to teach the app your first item</Text>;
  }
  if (!match) return null;
  const score = match.score.toFixed(2);
  return (
    <View style={styles.labelBox}>
      <Text style={[styles.labelText, match.item && { color: MATCH_COLOR }]}>
        {match.item ? match.item.label : 'Unknown item'}
      </Text>
      <Text style={styles.scoreText}>
        {match.item ? `similarity ${score}` : `closest: ${match.candidate?.label ?? '–'} (${score})`}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.three,
    padding: Spacing.four,
  },
  centeredText: {
    textAlign: 'center',
  },
  topBar: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    alignItems: 'center',
  },
  pill: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: Spacing.two,
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two,
    borderRadius: 999,
    backgroundColor: 'rgba(0,0,0,0.55)',
  },
  pillText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 600,
  },
  reticleArea: {
    ...StyleSheet.absoluteFill,
    alignItems: 'center',
    justifyContent: 'center',
    gap: Spacing.three,
  },
  reticle: {
    width: '72%',
    aspectRatio: 1,
    borderWidth: 3,
    borderRadius: Spacing.four,
  },
  labelBox: {
    alignItems: 'center',
    paddingHorizontal: Spacing.three,
    paddingVertical: Spacing.two,
    borderRadius: Spacing.three,
    backgroundColor: 'rgba(0,0,0,0.6)',
  },
  labelText: {
    color: '#fff',
    fontSize: 24,
    fontWeight: 700,
  },
  scoreText: {
    color: 'rgba(255,255,255,0.7)',
    fontSize: 12,
    fontVariant: ['tabular-nums'],
  },
  hint: {
    color: '#fff',
    fontSize: 15,
    textAlign: 'center',
    paddingHorizontal: Spacing.four,
    textShadowColor: 'rgba(0,0,0,0.8)',
    textShadowRadius: 4,
  },
  bottomBar: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    alignItems: 'center',
    paddingHorizontal: Spacing.three,
  },
  panel: {
    alignSelf: 'stretch',
    gap: Spacing.three,
    padding: Spacing.three,
    borderRadius: Spacing.four,
    backgroundColor: 'rgba(0,0,0,0.7)',
  },
  panelText: {
    color: '#fff',
    fontSize: 15,
    textAlign: 'center',
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: Spacing.three,
  },
  primaryButton: {
    minHeight: 48,
    paddingHorizontal: Spacing.four,
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#208AEF',
  },
  primaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 700,
  },
  secondaryButton: {
    minWidth: 80,
    minHeight: 44,
    alignItems: 'center',
    justifyContent: 'center',
  },
  secondaryButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 600,
  },
  shutter: {
    width: 72,
    height: 72,
    borderRadius: 36,
    borderWidth: 4,
    borderColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255,255,255,0.25)',
  },
  shutterText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 700,
    fontVariant: ['tabular-nums'],
  },
  input: {
    minHeight: 48,
    paddingHorizontal: Spacing.three,
    borderRadius: Spacing.three,
    color: '#fff',
    fontSize: 17,
    backgroundColor: 'rgba(255,255,255,0.15)',
  },
  disabled: {
    opacity: 0.4,
  },
});
