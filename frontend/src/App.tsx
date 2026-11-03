import { useRef, useState } from "react";
import { warmUp } from "./lib/api";
import { DemoMode } from "./modes/DemoMode";
import { PhonePage } from "./modes/PhonePage";
import { PhoneScreenMode } from "./modes/PhoneScreenMode";
import { WebcamMode } from "./modes/WebcamMode";

type Mode = "start" | "webcam" | "phone" | "demo";

export function App() {
  const phoneMatch = window.location.pathname.match(/^\/phone\/([\w-]+)/);
  const initial: Mode = new URLSearchParams(window.location.search).has("demo") ? "demo" : "start";
  const [mode, setMode] = useState<Mode>(initial);
  // Someone reached for a choice: wake the GPU services now so the session starts warm. Only on
  // a real touch (a visitor who never interacts wakes nothing), once per page load.
  const warmed = useRef(false);
  const warm = () => {
    if (warmed.current) return;
    warmed.current = true;
    warmUp();
  };
  if (phoneMatch) return <PhonePage sessionId={phoneMatch[1]} />;

  const exit = () => setMode("start");
  if (mode === "webcam") return <WebcamMode onExit={exit} />;
  if (mode === "phone") return <PhoneScreenMode onExit={exit} />;
  if (mode === "demo") return <DemoMode onExit={exit} />;

  return (
    <div className="start">
      <h1>
        Muna <b>Lens</b>
      </h1>
      <p className="lede">
        Solve on paper. Lens reads each step, finds the first real mistake, and asks you a question about it. It
        never gives the answer.
      </p>
      <div className="choices" onPointerEnter={warm} onFocus={warm} onTouchStart={warm}>
        <button onClick={() => setMode("webcam")}>
          <strong>Use this laptop’s camera</strong>
          <span>Hold your notebook up and point: “is this right?”</span>
        </button>
        <button onClick={() => setMode("phone")}>
          <strong>Use my phone as the camera</strong>
          <span>Scan a code, hold the phone over the page.</span>
        </button>
        <button className="quiet" onClick={() => setMode("demo")}>
          <strong>See a recorded example</strong>
          <span>No camera needed.</span>
        </button>
      </div>
    </div>
  );
}
