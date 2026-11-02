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
| Fallback brain when the shared Spark endpoint is slow on images | **Muse Glimmer** 30B (Apache 2.0) as `RedHatAI/Muse-Glimmer-30B-FP8-block` + DFlash draft head `meta-models/Muse-Glimmer-30B-assistant` | Self-hosted on **Modal**, one L40S, vLLM 0.28 OpenAI-compatible (`backend/modal_apps/glimmer.py`, app `muna-lens-glimmer`, served name `muse-glimmer`, bearer key `GLIMMER_API_KEY`). Switched to purely by settings: `SPARK_API_KEY`, `SPARK_BASE_URL`, `SPARK_MODEL=muse-glimmer` | Measured 2026-10-30, 13 EDU-CIRCUIT-HW pages one after another, warm: page read median 11.5 s, p90 13.1 s, max 13.7 s. Text reply first token 2.4–3.4 s (Spark: 1.3–1.6 s). Cold start 229 s. Reading similarity on 15 pages: mean 0.78, median 0.91 (Spark on the same pages: 0.80 / 0.87). L40S ≈ $1.95/h while warm (≈ $0.006 per page read when busy); scales to zero after 180 s idle. `reasoning_effort` is mapped onto Glimmer's "Reasoning strength" (`minimal` → `low`); low uses about half the tokens of high. |

Muse Spark Open gotchas:
- It is a reasoning model. Reasoning tokens count against `max_tokens`; with `max_tokens=300` it returned `None`. Use ≥1500 for replies, ≥3000 for page checks.
- It accepts audio (`{"type":"audio_url","audio_url":{"url":"data:audio/wav;base64,..."}}` and `input_audio` both work) and transcribes accurately, but a reply from audio starts after 4–5 s. Live speech goes through Omnilingual ASR → text instead.
- At low reasoning effort a hint leaked the step ("subtract 5 from one side…"). The guardrail (A4) is required, not optional.

Spark on Baseten stays the default. Re-test on 2026-10-30, 13 page reads over 11 minutes (`low` effort): 12 took 6.7–10.8 s (p90 10.8 s), and one queued 76 s before its first byte (81 s in total). Text replies took 1.4–1.7 s (the first, 3.3 s). The rule was "usable if p90 ≤ 15 s and no call over 45 s", so it failed on that one outlier. Glimmer is the self-hosted fallback. Reading quality is about even, and Glimmer never queues, but it is slower on text and costs GPU time while warm.

Muse Glimmer gotchas:
- On a 48 GB L40S, the FP8 weights plus the draft head take 37.3 GiB. At 32k context there was too little room left for the KV cache (0.52 GiB) and the engine failed to start. The app uses 16k context, `--gpu-memory-utilization 0.95` and 4 sequences, which leaves 1.87 GiB.
- vLLM prints its CLI args, so `--api-key` would put the key in the Modal logs. The key goes in through `VLLM_API_KEY` instead.
- The reasoning parser (`muse_glimmer`) sends reasoning into `reasoning_content`, so `content` holds only the answer and `response_format=json_object` works.

## Product decisions (settled)

- **Web app only.** Works with a **laptop webcam** (student holds the notebook up and points: "this line?") and a **phone camera** paired by QR code (phone over the notebook; laptop shows the screen and plays voice). Phone runs in the mobile browser; no native app. Meta glasses appear only on the vision slide (Ethiopia is not a supported country for the Wearables toolkit).
- **Proactive tutor.** It learns what the student is working on by asking and conversing, and confirms rather than assumes.
- **All STEM subjects.** No subject-specific builds; light prompt tuning only where testing demands it.
- **Live, interruptible voice conversation.** English only.
- **Minimal Lens screen**, designed fresh: live page, outline on the mistake, tutor pointer, list of read steps (ok / error). Exact look is open.
- **Cheap page checks.** Free in-browser checks (new ink appeared, hand out of the way) decide when to call Spark Open, plus a manual "check my work" button. Exact cadence and cost target are measured in testing.
- **Standalone architecture.** Lens runs as an independent, self-contained service.
- **Hosting:** app on the owner's AWS Lightsail under a subdomain; GPU models on Modal (free $30/month tier covers dev + testing).

- **Voice transport:** hand-built, no LiveKit. Browser loudness VAD cuts utterances and detects barge-in; WAV over the `/ws/tutor` WebSocket → Omnilingual ASR → Spark (`minimal`, streamed) → Kokoro sentence by sentence → PCM back. One less service to run; measured 1.4–1.8 s to first audio on a warm system.
- **SAM 3.1 tracks live; SAM 3 stays for still frames.** Live tracking runs `facebook/sam3.1` (`sam3.1_multiplex.pt`) in Meta's own `sam3` package on Modal (`modal_apps/sam31.py`, app `muna-lens-sam31`, one L40S). The package is built for whole videos; Lens feeds it a live stream one frame at a time (no look-ahead: the detector's prefetch would otherwise run on the previous frame) with one concept session for "hand" and one for "pen", sharing one image encoding per frame. The browser sends 512 px JPEGs (~15 kB) offered at 5 fps with at most 2 unanswered, over `/ws/track` (the app relays, so the endpoint and its token stay server-side). From the tracks, `lens/tracking.py` derives the phase (writing / paused / clear), the pointer (pen tip, else the fingertip reaching farthest in) and `line_done` (a writing spell ended and the hand left the region the pen tip covered), which triggers the page check. SAM 3 found no handwriting concept ("handwriting", "handwritten text", "equation", "text": 0 masks on real pages), so the outline follows the page by template matching the mistake line from the frame the check read (plumbing, not a model). The still-frame service (`muna-lens-sam`, SAM 3 via Transformers) still draws the outline mask and answers "this line?" when tracking is off; the pixel detector stays the trigger fallback.
- **Stack:** Vite + React + TypeScript frontend, FastAPI backend (Python, uv). Lens is a client-only camera app, so Vite fits better than Next.js; FastAPI serves the built frontend, so a deploy is one process. `make dev` runs both locally.

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

## Evaluation data

| Source | What it covers | License / access | Use |
|---|---|---|---|
| **FERMAT** `ai4bharat/FERMAT` (Hugging Face, ACL 2025) | 2,244 handwritten math solutions (grades 7–12) with planted errors labelled for detection, localisation and correction; error types computational, conceptual, notational, presentation | CC BY 4.0; click-through agreement on Hugging Face | **Main scorecard** for catch rate and false alarms |
| **EDU-CIRCUIT-HW** `gt-learning-innovation/CIRCUIT_EDU_HW_ACL` (GitHub, 2026) | 1,334 real undergraduate circuit-analysis solutions (with calculus, ODEs, physics) and expert transcriptions | No dataset license stated: private testing only, never redistributed | **Transcription accuracy** on university handwriting (A2) |
| **Own set** under `eval/own/` | 10–15 pages of real university calculus, physics and chemistry work, shot on phone and webcam (tilt, shadows, a hand in frame), some with a planted mistake | Writers' consent required | **Camera conditions and subjects** the datasets lack; demo honesty |

Rules:
- External datasets are **downloaded at run time** into `.cache/eval/` (gitignored), never committed. The repo carries only the loaders and harness.
- Own-set pages are committed only with each writer's consent; otherwise they live in the gitignored cache too.
- Report scores **per source**; a FERMAT score is not a camera-photo score.
- FERMAT and the own set measure error finding; EDU-CIRCUIT-HW measures reading only.

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
- **E0 Eval sets** (no deps, starts day 1, runs alongside everything). Three sources, each covering a gap the others leave (see *Evaluation data* below). *Done:* a loader yields every item from all three as `(image, problem, first_error_step | "none")`, and the own set has its first 10 pages.
- **A5 Eval harness** (A3, E0). Runs reader + checker over the eval sets; reports caught / missed / false alarms and cost per page, separately per source. *Done:* one command prints the scorecard. **Gate:** false alarms ≤ 1 in 20 correct pages before A4 work is tuned further; the scorecard is a demo asset.
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

### Build status

Update this table when a node moves. "Built offline" means the code and its offline tests exist, but the done criterion needs a key, a GPU service or real photos.

| Node | Status | What remains |
|---|---|---|
| A0 | Done | — |
| A1 | Done: live text, image and audio calls pass and log cost. Every call streams under the hood; slow image calls are hedged. `SPARK_API_KEY` (wins over `BASETEN`) + `SPARK_BASE_URL` + `SPARK_MODEL` point it at any Spark-compatible server (Glimmer on Modal verified); a server that rejects `reasoning_effort` is tolerated | Confirm Baseten's price (using 1.25 / 4.25 USD per M tokens from a third-party listing) |
| A2 | Measured: reading 15 real EDU-CIRCUIT-HW pages scores a median 0.93 similarity to expert transcriptions (mean 0.82; low scores are diagram-heavy pages). Boxes are asked as corners | A few pages read at 0.56–0.76; "every step on 10 pages" not yet met strictly |
| A3 | Done: SymPy checker (+ arithmetic inside wordy lines), whole-page model judge with a confirmation pass, re-read before a SymPy claim | Tune on more eval results |
| A4 | Done: 20 planted mistakes, hints at `low` effort, independent high-effort audit flagged 0 leaks | Make the more generic hints more specific |
| E0 | Loaders done; FERMAT downloaded; EDU-CIRCUIT-HW sample fetched by range requests | Own set: first 10 consented pages |
| A5 | Done: `make eval`. FERMAT (40 balanced pages): catch rate 9/18, false alarms 1/20 (gate passed, at the limit), $0.032 and 88 s per page | Re-run after the newest checker changes; more pages for tighter numbers |
| C1 | Done: SAM 3 still frames on Modal (masks for 11/11 reader step boxes, fingertip on 8/8 hand photos, 0.15 s warm, ~80 s cold). **SAM 3.1 live tracking** (`muna-lens-sam31`, L40S): hand + pen per frame 150–210 ms on the GPU (A10G: 380 ms, too slow); frame → result 400–420 ms median streaming from Ethiopia (3.3–3.7 fps processed); cold start 24–29 s; scales to zero ~60–80 s after the last session; ≈ $1.95/h while a session is open, $0 otherwise | Real-camera footage: on a real CC BY close-up calligraphy clip only part of the hand/pen was found (out of distribution); film own webcam/phone clips |
| C2 | Geometry done; pointing uses the hand mask (+ a held pen); right line picked in the live F1 session. Live: pen tip within 0.03 of truth on 96 % of frames (median 0.007) on a synthetic clip (real handwriting page, CC0 hand + pen, drifting camera); tracked pointer answers "this line?" when fresh. Outline follow: centre error median 0.003–0.006, max 0.010, starting from a check frame 2 s old | 10 real photos of a finger on a line; real-camera clips |
| D1 | Done: webcam + phone QR pairing (~0.3 s locally), change detection; a new line triggered exactly one upload in a fake-camera browser run. Live tracking owns the trigger: on the synthetic clip `line_done` fired once per hand-clear spell (3/3) and never while the hand covered the new line; fake-camera browser run: tracking live after a 29 s cold start, checks triggered by `line_done`, hint shown, outline moved with the drifting camera | Verify on a real phone over HTTPS |
| D2 | Done: renders a recorded `Page` + `Finding` at laptop and phone widths (`/?demo`) | — |
| V1 | Done: ASR 0.22–0.35 s for a 4.7 s utterance (exact transcript); TTS first audio ~0.25 s | — |
| V2 | Done: first audio 1.4–1.8 s after a spoken question (warm); barge-in stops speech; also verified in a browser with a fake mic | Real-mic test |
| F1 | Done: live session — correct line silent, wrong line outlined + spoken question, "is this line right?" with pointing, fix acknowledged; nothing spoken leaks the answer | Same on real handwriting with a real camera |
| F2 | Prepared: `deploy/setup-lightsail.sh` (systemd + Caddy HTTPS), optional `LENS_ACCESS_CODE` tester gate | Run on the Lightsail instance; login choice |
| F3 | Not started | Real students, after F2 |

Known issue: image calls on the shared Baseten endpoint sometimes queue for a minute or more (1 in 13 on 2026-10-30, 81 s; earlier runs saw 20–200 s). Text replies stay ~1.5 s. Fallback: self-hosted Muse Glimmer on Modal (see *Model stack*), with steady 8–14 s warm page reads. Deploy it with `make deploy-glimmer` and switch with the `SPARK_*` settings in `.env.example`.
