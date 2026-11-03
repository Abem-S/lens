import { useEffect, useRef, useState } from "react";
import { LensScreen } from "../components/LensScreen";
import { TutorBar } from "../components/TutorBar";
import { SessionActions } from "../components/SessionActions";
import { useCamera, useInterval } from "../lib/camera";
import { ActivityTracker, ChangeDetector, type DetectorState } from "../lib/changeDetect";
import { FrameSampler, grabJpeg } from "../lib/frames";
import { STATE_LABEL } from "../lib/detectorLabels";
import { RESTING_NOTE, useIdleMinutes, usePresence } from "../lib/presence";
import { type Box, useLensTracking } from "../lib/tracking";
import { useTutor } from "../lib/useTutor";

/** Laptop webcam: the student holds the notebook up to the camera. */
export function WebcamMode({ onExit }: { onExit: () => void }) {
  const { videoRef, error: camError, ready } = useCamera("user");
  const sampler = useRef<FrameSampler | null>(null);
  // Facing the student, the face and the hands holding the notebook are always in view,
  // and a hand-held page never sits perfectly still: skip the skin test, tolerate jitter.
  const detector = useRef(new ChangeDetector({ skinThreshold: 2, motionThreshold: 0.04 }));
  const [state, setState] = useState<DetectorState>("waiting");
  const activity = useRef(new ActivityTracker());
  // GPU work (tracking, warm voice) runs only while someone is here (lib/presence.ts).
  const presence = usePresence(useIdleMinutes());
  // The tutor gets the current frame with each question, for "this line?" pointing.
  const [trackId] = useState(() => `cam-${Math.random().toString(36).slice(2, 10)}`);
  const tutor = useTutor(async (side = 1024) => (videoRef.current ? grabJpeg(videoRef.current, side, side < 1024 ? 0.7 : 0.8) : null), trackId,
    presence.active, presence.interact);
  // Live SAM 3.1 tracking owns the check trigger when it is up; the detector below is the fallback.
  const track = useLensTracking(() => videoRef.current, ready && presence.active, tutor.result,
    (box) => capture("line_done", box), trackId);
  const video = videoRef.current;
  const aspect = video?.videoWidth ? video.videoWidth / video.videoHeight : 16 / 9;

  // A hand or pen in view is someone at work.
  const seenSeq = track.last?.objects.length ? track.last.seq : null;
  useEffect(() => {
    if (seenSeq !== null) presence.motion();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seenSeq]);

  const capture = async (why: string, box?: Box) => {
    const v = videoRef.current;
    if (!v || !presence.active) return;
    presence.interact();
    // The detector's baseline becomes this frame, so it does not check the same ink again.
    const f = (sampler.current ??= new FrameSampler()).sample(v);
    if (f) detector.current.markChecked(f);
    const blob = await grabJpeg(v);
    if (!blob) return;
    track.markCheck();
    tutor.check(blob, why, box);
  };

  useInterval(
    () => {
      const v = videoRef.current;
      if (!v) return;
      const f = (sampler.current ??= new FrameSampler()).sample(v);
      if (!f) return;
      const r = detector.current.push(f);
      setState(r.state);
      if (r.state === "moving" || r.state === "hand") presence.motion();
      // Without live tracking, the tutor's sense of the moment comes from the detector.
      const act = activity.current.report(r.state, performance.now());
      if (!track.live && act) tutor.activity(act);
      if (r.trigger && track.stillAllowed(r.trigger)) capture(r.trigger);
    },
    250,
    ready,
  );

  const waking = tutor.voiceOn && !tutor.voiceReady ? "Waking up Lens's voice…" : null;
  return (
    <LensScreen
      aspect={aspect}
      media={<video ref={videoRef} playsInline muted />}
      result={track.result}
      status={camError ? "Camera off" : tutor.note ?? waking ?? track.label ?? STATE_LABEL[state]}
      busy={tutor.busy}
      error={camError ?? tutor.error}
      notice={tutor.notice ?? (presence.active ? track.note : null)}
      resting={presence.active ? null : RESTING_NOTE[presence.state as "idle" | "hidden"]}
      onWake={presence.interact}
      onExit={onExit}
      pointingStep={tutor.pointing?.step_id ?? track.pointingStep}
      extra={
        <TutorBar state={tutor.state} captions={tutor.captions} voiceOn={tutor.voiceOn}
          voiceAvailable={tutor.voiceAvailable} latency={tutor.latency}
          onStart={tutor.startVoice} onStop={tutor.stopVoice}
          showCaptions={tutor.settings.captions} speaksAloud={tutor.settings.voice} onSend={tutor.sendText} />
      }
      actions={<SessionActions tutor={tutor} onExit={onExit} />}
      onReadAloud={tutor.readPage}
    />
  );
}
