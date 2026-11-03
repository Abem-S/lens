// Live video from the phone to the laptop over WebRTC, peer to peer. On the same Wi-Fi
// the video never leaves the room (~0.1 s behind, full frame rate); the JPEG preview
// through the server is only the fallback while this connects or when it cannot.
//
// Signalling rides on the existing pairing socket as {type: "rtc", ...} messages:
// the phone offers, the laptop answers; the laptop asks for a fresh offer with
// {type: "rtc_request"} whenever it (re)connects.

const CONFIG: RTCConfiguration = { iceServers: [{ urls: "stun:stun.l.google.com:19302" }] };

export type Signal = (msg: object) => void;
export type RtcMessage =
  | { type: "rtc"; sdp: RTCSessionDescriptionInit }
  | { type: "rtc"; candidate: RTCIceCandidateInit };

function wire(pc: RTCPeerConnection, signal: Signal, side: string) {
  pc.onicecandidate = (e) => e.candidate && signal({ type: "rtc", candidate: e.candidate.toJSON() });
  pc.oniceconnectionstatechange = () => console.info(`[lens] ${side} video link: ${pc.iceConnectionState}`);
}

/** Phone side: send the camera stream. Returns a handler for incoming rtc messages and a stop(). */
export function startSender(stream: MediaStream, signal: Signal, onState?: (s: RTCPeerConnectionState) => void) {
  const pc = new RTCPeerConnection(CONFIG);
  wire(pc, signal, "phone");
  pc.onconnectionstatechange = () => onState?.(pc.connectionState);
  for (const track of stream.getVideoTracks()) {
    const sender = pc.addTrack(track, stream);
    // Handwriting needs sharp frames more than smooth motion.
    const params = sender.getParameters();
    params.degradationPreference = "maintain-resolution";
    sender.setParameters(params).catch(() => undefined);
  }
  pc.createOffer()
    .then((offer) => pc.setLocalDescription(offer))
    .then(() => pc.localDescription && signal({ type: "rtc", sdp: pc.localDescription.toJSON() }))
    .catch(() => undefined);
  return {
    handle(msg: RtcMessage) {
      if ("sdp" in msg) pc.setRemoteDescription(msg.sdp).catch(() => undefined);
      else pc.addIceCandidate(msg.candidate).catch(() => undefined);
    },
    stop: () => pc.close(),
  };
}

/** Laptop side: answer the phone's offer and hand over the incoming stream. */
export function startReceiver(signal: Signal, onStream: (s: MediaStream | null) => void) {
  let pc: RTCPeerConnection | null = null;
  const reset = () => {
    pc?.close();
    const p = new RTCPeerConnection(CONFIG);
    pc = p;
    wire(p, signal, "laptop");
    // Show the video only once media actually flows; until then the JPEG preview stays up,
    // so a link that never connects never leaves the student looking at a black box.
    let incoming: MediaStream | null = null;
    p.ontrack = (e) => {
      incoming = e.streams[0] ?? new MediaStream([e.track]);
      if (p.connectionState === "connected") onStream(incoming);
    };
    p.onconnectionstatechange = () => {
      if (p.connectionState === "connected" && incoming) onStream(incoming);
      else if (["failed", "closed", "disconnected"].includes(p.connectionState)) onStream(null);
    };
    return p;
  };
  return {
    async handle(msg: RtcMessage) {
      if ("sdp" in msg && msg.sdp.type === "offer") {
        const p = reset(); // a new offer means the phone (re)started
        await p.setRemoteDescription(msg.sdp);
        const answer = await p.createAnswer();
        await p.setLocalDescription(answer);
        signal({ type: "rtc", sdp: p.localDescription!.toJSON() });
      } else if ("candidate" in msg && pc) {
        await pc.addIceCandidate(msg.candidate).catch(() => undefined);
      }
    },
    stop: () => pc?.close(),
  };
}
