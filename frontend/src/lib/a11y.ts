import { useCallback, useState } from "react";

/** Accessibility settings, remembered per browser (never sent anywhere but the tutor socket). */
export interface A11ySettings {
  captions: boolean; // show what Lens says and what it heard
  voice: boolean; // Lens speaks aloud (off: captions only, replies still come as text)
  speed: number; // speaking speed, 1 = normal
}

// Captions are on by default: they cost nothing, help in a noisy room or with an unfamiliar
// accent, and a student who cannot hear Lens should not have to find a setting to learn that
// Lens is talking to them.
export const DEFAULT_A11Y: A11ySettings = { captions: true, voice: true, speed: 1 };
export const SPEED_MIN = 0.7;
export const SPEED_MAX = 1.4;
const KEY = "lens.a11y.v1";

export function clampSpeed(v: number): number {
  if (!Number.isFinite(v)) return 1;
  return Math.round(Math.min(SPEED_MAX, Math.max(SPEED_MIN, v)) * 100) / 100;
}

export function loadA11y(): A11ySettings {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return DEFAULT_A11Y;
    const v = JSON.parse(raw) as Partial<A11ySettings>;
    return {
      captions: typeof v.captions === "boolean" ? v.captions : DEFAULT_A11Y.captions,
      voice: typeof v.voice === "boolean" ? v.voice : DEFAULT_A11Y.voice,
      speed: typeof v.speed === "number" ? clampSpeed(v.speed) : DEFAULT_A11Y.speed,
    };
  } catch {
    return DEFAULT_A11Y; // private window, blocked storage, bad JSON
  }
}

export function saveA11y(s: A11ySettings): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    /* storage unavailable: the setting lasts for this visit only */
  }
}

export function useA11ySettings(): [A11ySettings, (patch: Partial<A11ySettings>) => void] {
  const [settings, setSettings] = useState<A11ySettings>(loadA11y);
  const update = useCallback((patch: Partial<A11ySettings>) => {
    setSettings((s) => {
      const next = { ...s, ...patch, speed: clampSpeed(patch.speed ?? s.speed) };
      saveA11y(next);
      return next;
    });
  }, []);
  return [settings, update];
}
