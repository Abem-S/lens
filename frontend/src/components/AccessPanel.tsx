import { useEffect, useId, useRef, useState } from "react";
import { SPEED_MAX, SPEED_MIN, type A11ySettings } from "../lib/a11y";

interface Props {
  settings: A11ySettings;
  onChange: (patch: Partial<A11ySettings>) => void;
  voiceAvailable?: boolean;
}

/** One small panel of accessibility settings, opened from the top bar. */
export function AccessPanel({ settings, onChange, voiceAvailable = true }: Props) {
  const [open, setOpen] = useState(false);
  const id = useId();
  const button = useRef<HTMLButtonElement | null>(null);
  const panel = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    panel.current?.querySelector<HTMLElement>("input")?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setOpen(false);
        button.current?.focus();
      }
    };
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!panel.current?.contains(t) && !button.current?.contains(t)) setOpen(false);
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("pointerdown", onDown);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("pointerdown", onDown);
    };
  }, [open]);

  return (
    <div className="access">
      <button ref={button} className="bar-btn" aria-label="Accessibility settings" aria-expanded={open}
        aria-controls={`${id}-panel`}
        onClick={() => setOpen((o) => !o)}>
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor"
          strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="4.5" r="1.8" />
          <path d="M4.5 8.5 12 10l7.5-1.5M12 10v4.5M12 14.5 8.5 21M12 14.5l3.5 6.5" />
        </svg>
        <span className="bar-btn-label">Accessibility</span>
      </button>
      {open && (
        <div ref={panel} id={`${id}-panel`} className="access-panel" role="group" aria-label="Accessibility settings">
          <h2>Accessibility</h2>
          <label className="toggle">
            <input type="checkbox" checked={settings.captions} onChange={(e) => onChange({ captions: e.target.checked })} />
            <span>
              <b>Captions</b>
              <small>Show what Lens says and what it heard.</small>
            </span>
          </label>
          <label className="toggle">
            <input type="checkbox" checked={settings.voice} disabled={!voiceAvailable}
              onChange={(e) => onChange({ voice: e.target.checked })} />
            <span>
              <b>Lens speaks aloud</b>
              <small>{voiceAvailable ? "Off: Lens replies in captions only." : "Voice is not available right now."}</small>
            </span>
          </label>
          <label className="speed" htmlFor={`${id}-speed`}>
            <span>
              <b>Speaking speed</b>
              <output htmlFor={`${id}-speed`}>{settings.speed.toFixed(1)}×</output>
            </span>
            <input id={`${id}-speed`} type="range" min={SPEED_MIN} max={SPEED_MAX} step={0.1} value={settings.speed}
              disabled={!settings.voice || !voiceAvailable}
              aria-valuetext={`${settings.speed.toFixed(1)} times normal speed`}
              onChange={(e) => onChange({ speed: Number(e.target.value) })} />
            <span className="speed-ends" aria-hidden="true">
              <small>Slower</small>
              <small>Faster</small>
            </span>
          </label>
          <p className="access-note muted">You can also type to Lens, and ask it to read your page aloud.</p>
        </div>
      )}
    </div>
  );
}
