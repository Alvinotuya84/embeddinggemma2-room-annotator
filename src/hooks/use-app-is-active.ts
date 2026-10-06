import { useEffect, useState } from 'react';
import { AppState } from 'react-native';

/** Whether the app is in the foreground. Used to stop the camera when backgrounded. */
export function useAppIsActive() {
  const [isActive, setIsActive] = useState(AppState.currentState === 'active');

  useEffect(() => {
    const subscription = AppState.addEventListener('change', (state) => {
      setIsActive(state === 'active');
    });
    return () => subscription.remove();
  }, []);

  return isActive;
}
