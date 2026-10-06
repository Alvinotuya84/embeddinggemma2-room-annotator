import { useMemo } from 'react';
import { useImageEmbedder } from 'react-native-executorch';
import type { ImageBuffer } from 'react-native-executorch/cv';
import { useFrameOutput } from 'react-native-vision-camera';
import { type GPUFrame, useResizer } from 'react-native-vision-camera-resizer';
import { createSynchronizable, scheduleOnRN } from 'react-native-worklets';

import { CROP_SIZE, IMAGE_EMBEDDER, SCAN_INTERVAL_MS } from './model-config';
import type { CapturedSample } from './types';

type Options = {
  /** Called about every `SCAN_INTERVAL_MS` with the embedding of the centered crop. */
  onEmbedding: (vector: Float32Array) => void;
  /** Called once per `requestCapture()` with the embedding and the crop it came from. */
  onCapture: (sample: CapturedSample) => void;
};

/**
 * Embeds the centered square of the live camera feed on the frame-processor
 * thread. The GPU resizer crops and converts each frame to RGB, and the
 * embedding runs synchronously in the worklet, so the JS thread only receives
 * finished vectors.
 */
export function useFrameEmbedder({ onEmbedding, onCapture }: Options) {
  const embedder = useImageEmbedder(IMAGE_EMBEDDER);
  const { embedWorklet } = embedder;

  const { resizer, error: resizerError } = useResizer({
    width: CROP_SIZE,
    height: CROP_SIZE,
    channelOrder: 'rgb',
    dataType: 'uint8',
    scaleMode: 'cover',
    pixelLayout: 'interleaved',
  });

  // Shared between the JS thread and the frame-processor worklet runtime.
  const lastRunAt = useMemo(() => createSynchronizable(0), []);
  const captureRequested = useMemo(() => createSynchronizable(false), []);

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    dropFramesWhileBusy: true,
    onFrame(frame) {
      'worklet';
      const capture = captureRequested.getDirty();
      const now = Date.now();
      if (!resizer || !embedWorklet || (!capture && now - lastRunAt.getDirty() < SCAN_INTERVAL_MS)) {
        frame.dispose();
        return;
      }
      lastRunAt.setBlocking(now);
      if (capture) captureRequested.setBlocking(false);

      let resized: GPUFrame | undefined;
      try {
        resized = resizer.resize(frame);
        const data = new Uint8Array(resized.getPixelBuffer());
        const input: ImageBuffer = { data, width: CROP_SIZE, height: CROP_SIZE, format: 'rgb', layout: 'hwc' };
        const vector = embedWorklet(input);
        if (capture) {
          // Copy the pixels: the GPU frame's buffer is released in `finally`.
          scheduleOnRN(onCapture, { vector, pixels: data.slice() });
        } else {
          scheduleOnRN(onEmbedding, vector);
        }
      } catch {
        // Frames can be invalidated mid-flight when the camera stops or unmounts.
        if (capture) captureRequested.setBlocking(true);
      } finally {
        resized?.dispose();
        frame.dispose();
      }
    },
  });

  return {
    frameOutput,
    requestCapture: () => captureRequested.setBlocking(true),
    isReady: embedder.isReady && resizer != null,
    downloadProgress: embedder.downloadProgress,
    error: embedder.error ?? resizerError,
  };
}
