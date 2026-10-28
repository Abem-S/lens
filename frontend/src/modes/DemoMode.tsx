import { LensScreen } from "../components/LensScreen";
import { DEMO_ASPECT, DEMO_RESULT, demoPageSvg } from "../demo/fixture";

/** D2 check: a recorded Page + Finding rendered on the Lens screen. */
export function DemoMode({ onExit }: { onExit: () => void }) {
  return (
    <LensScreen
      aspect={DEMO_ASPECT}
      media={<img src={demoPageSvg()} alt="A recorded notebook page" />}
      result={DEMO_RESULT}
      status="Demo: a recorded page"
      onExit={onExit}
    />
  );
}
