import { useState } from "react";
import { DemoMode } from "./modes/DemoMode";
import { PhonePage } from "./modes/PhonePage";
import { PhoneScreenMode } from "./modes/PhoneScreenMode";
import { WebcamMode } from "./modes/WebcamMode";

type Mode = "start" | "webcam" | "phone" | "demo";

export function App() {
  const phoneMatch = window.location.pathname.match(/^\/phone\/([\w-]+)/);
  const initial: Mode = new URLSearchParams(window.location.search).has("demo") ? "demo" : "start";
  const [mode, setMode] = useState<Mode>(initial);
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
        Solve on paper. Lens reads each line, finds the first real mistake, and asks you a question about it. It
        never gives the answer.
      </p>
      <div className="choices">
        <button onClick={() => setMode("webcam")}>
          <strong>Use this laptop’s camera</strong>
          <span>Hold your notebook up and point: “this line?”</span>
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
