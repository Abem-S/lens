import { useRef, useState } from "react";
import { LensScreen } from "../components/LensScreen";
import { useCamera, useInterval } from "../lib/camera";
import { ChangeDetector, type DetectorState } from "../lib/changeDetect";
import { FrameSampler, grabJpeg } from "../lib/frames";
import { STATE_LABEL, useChecker } from "../lib/useChecker";

/** Laptop webcam: the student holds the notebook up to the camera. */
export function WebcamMode({ onExit }: { onExit: () => void }) {
  const { videoRef, error: camError, ready } = useCamera("user");
  const sampler = useRef<FrameSampler | null>(null);
  const detector = useRef(new ChangeDetector());
  const [state, setState] = useState<DetectorState>("waiting");
  const checker = useChecker();
  const video = videoRef.current;
  const aspect = video?.videoWidth ? video.videoWidth / video.videoHeight : 16 / 9;

  const capture = async (why: Parameters<typeof checker.run>[1]) => {
    const v = videoRef.current;
    if (!v) return;
    const blob = await grabJpeg(v);
    if (!blob) return;
    if (why === "manual") {
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (f) detector.current.markChecked(f);
    }
    checker.run(blob, why);
  };

  useInterval(
    () => {
      const v = videoRef.current;
      if (!v) return;
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (!f) return;
      const r = detector.current.push(f);
      setState(r.state);
      if (r.trigger) capture(r.trigger);
    },
    250,
    ready,
  );

  return (
    <LensScreen
      aspect={aspect}
      media={<video ref={videoRef} playsInline muted />}
      result={checker.result}
      status={camError ?? checker.note ?? STATE_LABEL[state]}
      busy={checker.busy}
      error={checker.error}
      onCheck={() => capture("manual")}
      onExit={onExit}
    />
  );
}
