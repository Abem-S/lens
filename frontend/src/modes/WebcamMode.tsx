import { useRef, useState } from "react";
import { LensScreen } from "../components/LensScreen";
import { TutorBar } from "../components/TutorBar";
import { useCamera, useInterval } from "../lib/camera";
import { ChangeDetector, type DetectorState } from "../lib/changeDetect";
import { FrameSampler, grabJpeg } from "../lib/frames";
import { STATE_LABEL } from "../lib/detectorLabels";
import { useTutor } from "../lib/useTutor";

/** Laptop webcam: the student holds the notebook up to the camera. */
export function WebcamMode({ onExit }: { onExit: () => void }) {
  const { videoRef, error: camError, ready } = useCamera("user");
  const sampler = useRef<FrameSampler | null>(null);
  const detector = useRef(new ChangeDetector());
  const [state, setState] = useState<DetectorState>("waiting");
  // The tutor gets the current frame with each question, for "this line?" pointing.
  const tutor = useTutor(async () => (videoRef.current ? grabJpeg(videoRef.current, 1024, 0.8) : null));
  const video = videoRef.current;
  const aspect = video?.videoWidth ? video.videoWidth / video.videoHeight : 16 / 9;

  const capture = async (why: string) => {
    const v = videoRef.current;
    if (!v) return;
    const blob = await grabJpeg(v);
    if (!blob) return;
    if (why === "manual") {
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (f) detector.current.markChecked(f);
    }
    tutor.check(blob, why);
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
      result={tutor.result}
      status={camError ?? tutor.note ?? STATE_LABEL[state]}
      busy={tutor.busy}
      error={tutor.error}
      onCheck={() => capture("manual")}
      onExit={onExit}
      pointingStep={tutor.pointing?.step_id}
      extra={
        <TutorBar state={tutor.state} captions={tutor.captions} voiceOn={tutor.voiceOn}
          voiceAvailable={tutor.voiceAvailable} latency={tutor.latency}
          onStart={tutor.startVoice} onStop={tutor.stopVoice} />
      }
    />
  );
}
