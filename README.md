# Muna Lens

A tutor that watches a student solve STEM problems on paper. A camera sees the
page; Lens reads each handwritten line, finds the **first real mistake**, outlines
it, and asks one Socratic question. It never gives the answer.

Intelligence runs on Meta open
models (Muse Spark Open, SAM 3, Omnilingual ASR); see [AGENTS.md](AGENTS.md) for
the plan, rules and build order.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

```sh
cp .env.example .env      # BASETEN, HF_TOKEN, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET
make install
make deploy-gpu           # once: SAM + voice services on Modal (scale to zero when idle)
make dev                  # open http://localhost:5173
```

What each key turns on:

| Key | Enables |
|---|---|
| `BASETEN` | Muse Spark Open: page reading, checking, hints, conversation |
| `MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET` | Voice (Omnilingual ASR + Kokoro), SAM outlines and "this line?" pointing |
| `HF_TOKEN` | SAM weights (at deploy) and the FERMAT eval set |

Without Modal the app still reads, checks and shows hints as text (`LENS_USE_MODAL=0`).
Press **Talk with Lens** to start the spoken conversation; talk over Lens at any time
and it stops.

- **No key yet?** Choose *See a recorded example* on the start screen (or open
  `/?demo`). Cameras, pairing and change detection all work without a key; page
  checks answer "BASETEN is not set".
- **Phone as the camera:** phone browsers only open the camera on HTTPS. Run
  `make dev-https`, open `https://<laptop-ip>:5173` on the laptop, choose
  *Use my phone*, and scan the code (accept the self-signed certificate once).
- **One process (deploy):** `make serve` builds the frontend and serves it from
  the backend on port 8000. Set `PUBLIC_URL` so the QR code points at it.

## Test

```sh
make test        # backend (pytest) + frontend (vitest); offline except one live Spark test when BASETEN is set
make eval        # A5 scorecard per source; needs BASETEN (FERMAT also needs HF_TOKEN)
```

Live checks for the GPU and voice nodes (they cost a little GPU time), run from `backend/`:

```sh
uv run --extra gpu --extra eval python scripts/c1_check.py   # SAM: step-box masks, fingertips, latency
uv run python scripts/make_track_clip.py PAGE.png HAND.png CLIP/ --drift   # synthetic tracking clip + truth
uv run --extra gpu python scripts/track_check.py CLIP/        # SAM 3.1 live tracking: latency, events, outline follow
uv run --extra gpu python scripts/v1_check.py                # ASR + TTS round trip, latency
uv run --extra gpu python scripts/v2_check.py                # needs `make serve`: first-audio latency, barge-in
uv run python scripts/a4_check.py                            # hints on planted mistakes + independent leak audit
uv run --extra gpu python scripts/f1_check.py                # needs `make serve`: a full tutor session
```

## Layout

```
backend/src/lens/
  contract.py     Page / Step / Finding (the shared contract)
  spark.py        A1  Muse Spark Open client (text, image, audio, streaming, cost log)
  reader.py       A2  frame -> Page
  mathparse.py    A3  one line of LaTeX -> SymPy, or a refusal
  checker.py      A3  does each line follow? first verified error
  judge.py        A3  whole-page model judge (+ confirmation) for lines SymPy cannot parse
  pipeline.py         read -> check -> re-read to confirm -> hint -> outline (shared by API, tutor, eval)
  hints.py        A4  Socratic hint writer + answer-leak guardrail
  pointing.py     C2  fingertip -> step, outlines
  sam.py          C1  client for the SAM still-frame service on Modal (modal_apps/sam.py)
  tracking.py     C1/C2 live SAM 3.1 tracking (modal_apps/sam31.py): relay, writing / pointing / line_done events
  voice.py        V1  client for ASR + TTS on Modal (modal_apps/voice.py)
  tutor.py        F1/V2 tutor session: conversation, spoken hints, barge-in, pointing
  eval/           E0 loaders, A5 harness (`lens-eval`)
  app.py          API, phone pairing + frame relay, serves the built frontend
frontend/src/
  lib/tracking.ts       D1  live tracking client: checks on line_done, outlines follow the page
  lib/changeDetect.ts   D1  new-ink / hand-clear detection (fallback trigger when tracking is off)
  lib/vad.ts, audio.ts  V2  speech detection, mic capture, playback with barge-in
  lib/useTutor.ts       F1  the tutor session in the browser
  modes/                D1  webcam, phone (QR pairing + capture), demo
  components/           D2  Lens screen: page, outline, tutor pointer, steps panel
eval/own/               own eval set labels (images only with the writer's consent)
```
