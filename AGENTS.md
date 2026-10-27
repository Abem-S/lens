# Muna Lens

A tutor that watches a student solve STEM problems on paper. A camera sees the page; Lens reads each handwritten step, finds the **first real mistake**, outlines it, and asks the student one Socratic question by voice. It never gives the answer.

Built for every student who studies on paper.

## Hard rules

1. **Secrets stay in `.env`** (gitignored). Commit only `.env.example` with variable names.
2. **Meta models do the intelligence**: seeing, reading, checking, talking, pointing, listening. Non-Meta parts are limited to voice output (Kokoro) and plumbing.
3. **A claimed mistake is a verified mistake.** When a step is algebra or calculus, SymPy must confirm the error before the tutor mentions it. A false alarm costs more than a missed error.
4. **Hints carry no answer.** A hint asks a question or points at a place; it never states the corrected step or the result.

## Model stack (decided)

| Job | Model | Where | Verified facts |
|---|---|---|---|
| See page, read steps, check, write hints, converse | **Muse Spark Open** `meta-models/muse-spark-open-1026` (use exactly this ID) | Baseten, OpenAI-compatible, `base_url=https://inference.baseten.co/v1`, key in `.env` as `BASETEN` | Accepts images (`image_url` data URL). Reads a page and found a planted error in ~5 s for ~$0.003. Text reply first token: ~1.2 s at `reasoning_effort="minimal"`, ~2 s low, ~3 s default. |
| Locate the wrong region (mask) and where the student points (fingertip / pen tip) | **SAM 3.1** `facebook/sam3.1` (backup `facebook/sam3`, which has `model.safetensors`) | Self-hosted on **Modal** GPU (~16 GB VRAM) | Access approved on Hugging Face for both repos. Text prompt is a short noun phrase → box + mask per match. ~30–60 s cold start: keep warm during sessions. |
| Speech to text | **Omnilingual ASR** `facebook/omniASR-CTC-1B` (backup `omniASR-CTC-300M`) | Modal | Not gated. Apache 2.0. ~2 GB VRAM for 300M. Supports Amharic (future). |
| Speech output | **Kokoro-82M** | Modal or CPU | Meta's TTS models (MMS-TTS, SeamlessM4T) are CC BY-NC: unusable for a startup. |

Muse Spark Open gotchas:
- It is a reasoning model. Reasoning tokens count against `max_tokens`; with `max_tokens=300` it returned `None`. Use ≥1500 for replies, ≥3000 for page checks.
- It accepts audio (`{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}` and `input_audio` both work) and transcribes accurately, but a reply from audio starts after 4–5 s. Live speech goes through Omnilingual ASR → text instead.
- At low reasoning effort a hint leaked the step ("subtract 5 from one side…"). The guardrail (A4) is required, not optional.

Muse Glimmer is out of scope: it adds nothing over Spark Open.

## Product decisions (settled)

- **Web app only.** Works with a **laptop webcam** (student holds the notebook up and points: "this line?") and a **phone camera** paired by QR code (phone over the notebook; laptop shows the screen and plays voice). Phone runs in the mobile browser; no native app. Meta glasses appear only on the vision slide (Ethiopia is not a supported country for the Wearables toolkit).
- **Proactive tutor.** It learns what the student is working on by asking and conversing, and confirms rather than assumes.
- **All STEM subjects.** No subject-specific builds; light prompt tuning only where testing demands it.
- **Live, interruptible voice conversation.** English only.
- **Minimal Lens screen**, designed fresh: live page, outline on the mistake, tutor pointer, list of read steps (ok / error). Exact look is open.
- **Cheap page checks.** Free in-browser checks (new ink appeared, hand out of the way) decide when to call Spark Open, plus a manual "check my work" button. Exact cadence and cost target are measured in testing.
- **Standalone architecture.** Lens runs as an independent, self-contained service.
- **Hosting:** app on the owner's AWS Lightsail under a subdomain; GPU models on Modal (free $30/month tier covers dev + testing).

## Shared contract

Every lane codes against this shape so lanes can build in parallel. Change it in one place: here.

```
Page {
  problem: string | null          # what is being solved, if known
  steps: Step[]                   # in reading order
}
Step {
  id: string
  latex: string                   # transcription
  bbox: [x, y, w, h]              # normalised 0–1 on the frame
  status: "unchecked" | "ok" | "error" | "unverifiable"
  verified_by: "sympy" | "model" | null
}
Finding {
  step_id: string                 # first erroneous step
  kind: string                    # e.g. "sign error moving a term across ="
  mask: polygon | null            # from SAM
  hint: string                    # Socratic, answer-free
}
```

## Build order (DAG)

Each node lists what it depends on and its **done** criterion. Nodes in different lanes with satisfied dependencies are built in parallel. Every node ships with a test that proves its done criterion.

```
A0 ─┬─ A1 ─┬─ A2 ── A3 ──┬── A5 ── A4 ──┐
    │      │             │              │
    │      │      E0 ────┘ (A5 = eval)  │
    │      │                            ├── F1 ── F2 ── F3
    ├─ C1 ─┴─ C2 ───────────────────────┤
    │                                   │
    ├─ D1 ── D2 ────────────────────────┤
    │                                   │
    └─ V1 ── V2 ────────────────────────┘
```

### Lane 0: foundation
- **A0 Repo skeleton.** `.gitignore` (covers `.env`), `.env.example`, frontend + backend scaffolds, one command to run locally. *Done:* app boots locally; `git status` shows no secret files.

### Lane A: brain (Muse Spark Open)
- **A1 Spark client** (A0). Text, image and audio inputs, streaming, `reasoning_effort` per call, retries, token/cost logging per call. *Done:* one call of each input type passes a test and logs its cost.
- **A2 Page reader** (A1). Frame → `Page` (steps in order, LaTeX, bbox, problem if visible). *Done:* on 10 eval pages, every step is transcribed and boxed in the right order.
- **A3 Step checker** (A2). For each consecutive pair, SymPy checks the step follows; non-algebraic steps go to Spark and are marked `verified_by: "model"`. Output: first erroneous step. *Done:* passes unit tests with planted algebra, calculus and sign errors, and marks correct derivations all `ok`.
- **E0 Eval set** (no deps, starts day 1, runs alongside everything). 30–50 real handwritten solutions across calculus, physics and chemistry, each labelled with its first erroneous step or "none". Include phone and webcam photos, messy handwriting, bad lighting. *Done:* labelled set committed under `eval/` (only with the writers' consent).
- **A5 Eval harness** (A3, E0). Runs reader + checker over the eval set; reports caught / missed / false alarms and cost per page. *Done:* one command prints the scorecard. **Gate:** false alarms ≤ 1 in 20 correct pages before A4 work is tuned further; the scorecard is a demo asset.
- **A4 Hint writer + guardrail** (A3). Writes one Socratic hint for a `Finding`; a second check rejects any hint that contains the corrected step, the result or an equivalent expression (SymPy compare against the expected next line), then regenerates. *Done:* across the eval errors, zero hints leak the answer.

### Lane C: SAM 3.1
- **C1 SAM service on Modal** (A0). Endpoint: image + (box or noun-phrase prompt) → masks. Warm-keeping switch for sessions. *Done:* returns a mask for a step bbox and for "fingertip" on test photos; warm latency measured and logged.
- **C2 Pointing + outlines** (C1, A2). Fingertip/pen-tip location → nearest `Step`; step bbox → clean outline mask for the `Finding`. *Done:* on 10 photos of a finger on a line, the right step is picked ≥ 9 times.

### Lane D: capture + screen
- **D1 Capture client** (A0). Webcam mode and phone mode (QR pairing; phone browser streams frames; screen wake lock). Change detection: send a frame only when new ink appears and the hand is clear; manual "check" button. *Done:* phone pairs by QR in under 10 s; writing a new line triggers exactly one frame upload.
- **D2 Lens screen** (D1, contract). Live page, outline from `Finding.mask`, tutor pointer, steps panel (ok / error). Fresh design. *Done:* renders a recorded `Page` + `Finding` correctly on laptop and phone widths.

### Lane V: voice
- **V1 ASR + TTS services** (A0). Omnilingual ASR and Kokoro on Modal. *Done:* a 5 s utterance transcribes and a sentence synthesises; both latencies logged.
- **V2 Voice loop** (V1, A1; custom WebSocket voice loop with VAD, interruption, and streamed ASR/TTS). Voice activity detection, interruption (student talks → tutor stops), ASR → Spark (`minimal` effort) → TTS. *Done:* spoken question to first audio reply under ~2.5 s on a warm system; barge-in stops speech.

### Lane F: integration (after lanes converge)
- **F1 Tutor agent** (A5, A4, C2, D2, V2). Proactive opening (asks what they are working on), page-change → read → check → if a verified `Finding`, outline it and speak the hint; answers "this line?" questions using pointing. *Done:* a full session on a real problem: the student writes a wrong line, Lens outlines it and asks a question; the student fixes it and Lens moves on, without leaking the answer.
- **F2 Deploy** (F1). Production deploy on Modal (`make deploy-web`) + GPU services; access control configured. *Done:* an outside tester completes F1's session from their own device via the URL.
- **F3 Real-student testing + cost** (F2). Several real students, own homework; measure cost per hour and error-catch rate in the wild; record consented clips. *Done:* cost/hour and scorecard numbers written down; at least one usable clip.

