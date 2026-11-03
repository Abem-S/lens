# Muna Lens

**A tutor that watches you solve maths and science problems on paper.** Prop a phone
over your notebook (or hold the page up to your laptop's webcam) and work as usual.
When you finish a step, Lens reads it, checks it, and if it is wrong, outlines it on
the screen and asks you one Socratic question out loud. It never gives the answer.

Built for every student who studies on paper. Students learn and sit exams with pen
and paper, but AI tutors only see typed text, and a wrong sign in step 2 of a
derivation goes unnoticed until the exam. Lens brings the tutor to the page.

## How it works

Meta open models do the intelligence; everything else is plumbing.

| Job | Model | Where |
|---|---|---|
| Watch the hand and pen live: when a step is finished, where the student points | **SAM 3.1** (`facebook/sam3.1`) | Modal GPU |
| Outline the mistake on a still frame; "is this step right?" pointing when live tracking is off | **SAM 3** (`facebook/sam3`) | Modal GPU |
| Read the handwriting, judge steps SymPy cannot, write the hint, hold the conversation | **Muse Spark Open** (`meta-models/muse-spark-open-1026`) | Baseten |
| Self-hosted fallback brain for demos | **Muse Glimmer 30B** | Modal GPU, vLLM |
| Hear the student | **Omnilingual ASR** (`facebook/omniASR-CTC-1B`) | Modal GPU |
| Speak | Kokoro-82M (Meta's own TTS models are non-commercial) | Modal GPU |

1. **Live video.** The phone streams its camera straight to the laptop over WebRTC.
   SAM 3.1 tracks the student's hand and pen a few times a second and decides when a
   step is finished (the pen stopped and the hand moved away). SAM does not read or
   segment handwriting: it found no handwriting concept on real pages, so reading is
   Muse Spark's job and the step boxes come from where the ink is.
2. **Read only the new step.** Lens finds the finished step's line in the frame and
   reads just that line, twice in parallel.
3. **Verify before speaking.** Algebra and calculus are checked with SymPy; steps it
   cannot parse go to a Muse Spark judge with a confirmation pass. A mistake is
   claimed only when independent readings agree on what is written. A false alarm
   costs more than a missed mistake.
4. **Hint without the answer.** Muse Spark writes one short question; a guardrail
   rejects any hint that contains the corrected step or the result (SymPy compares
   against the expected line) before it is spoken.
5. **Talk.** A live voice conversation you can interrupt. Lens also speaks up by itself,
   sparingly: a nudge when you have been stuck for a while, encouragement after a hard
   step, a remark when you cross something out.

The outline stays on the mistake as the page moves by matching the image of that line
from the frame Lens checked (template matching, not a model).

**Accessibility:** live captions of everything Lens says and hears (on by default),
type instead of talk, voice speed and a captions-only mode, "read my page aloud" in
spoken maths, and an end-of-session summary you can download. Nothing is stored
server-side: no accounts, no memory between sessions.

## How well it works

Measured on held-out pages of [FERMAT](https://huggingface.co/datasets/ai4bharat/FERMAT)
(handwritten maths solutions with planted mistakes, CC BY 4.0), 30 pages with a mistake
and 30 without, pages never used for tuning:

| Held-out set | Mistakes caught | False alarms on correct pages | Time per page | Model cost per page |
|---|---|---|---|---|
| Seed 8 | 17 / 30 | 1 / 27 | 22 s | $0.024 |
| Seed 9 | 18 / 30 | 1 / 30 | 19 s | $0.023 |

The target was at most 1 false alarm in 20 correct pages; both sets meet it. Lens is
built to stay quiet when unsure, so it misses about 4 in 10 mistakes rather than
flagging correct work. Seed 8 counts 27 correct pages because three pages FERMAT labels
correct really contain the mistake (copied from its own reference solution); they are
listed with reasons in `backend/src/lens/eval/loaders.py`.

- From the end of a finished wrong step to the spoken question: about 6–10 s (most of
  it the model reading and writing the hint).
- From the end of the student's speech to Lens's first word: about 1.8–2 s, measured
  over a remote network connection.
- Hints: 0 leaks of the answer on 20 planted mistakes, checked by an independent audit.
- Estimated running cost for one student alone: about $4 per hour (mostly GPUs kept
  warm; students using it at the same time share them). Every GPU service scales to
  zero a minute or two after the last student leaves.

All numbers above are from synthetic camera video and public datasets; tests with real
students are in progress. See [AGENTS.md](AGENTS.md) for the full plan, rules, build
order and every measurement.

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
- **Hosted on Modal (easiest for a phone):** `make deploy-web` builds the frontend and
  deploys the app with real HTTPS; it prints the URL. Open it on the laptop, choose
  *Use my phone* and scan the code from any network. Keys go to Modal as a secret
  from `.env`.
- **One process (deploy):** `make serve` builds the frontend and serves it from
  the backend on port 8000. Set `PUBLIC_URL` so the QR code points at it.

## Running it in public: cost, limits, slow model

- **GPU only while someone uses it.** No Modal app pins a container up. The tutor
  pings the voice and SAM services only while the browser says a student is active
  (it rests after `LENS_IDLE_MIN`, default 5 min, with no camera movement, speech or
  tap, at once when the tab is hidden, and after 4x that with no tap, speech or check
  even if the camera keeps seeing movement). Live tracking closes when Lens rests.
  After the last student leaves, each GPU app scales to zero on its own `scaledown_window`
  (voice and SAM 90 s after the last ping, SAM 3.1 60 s after its socket closes).
- **Limits** (all in `.env.example`, `0` = off): sessions open at once, tracking
  sessions at once, session starts per IP per hour, checks and spoken turns per session
  per minute, and a daily cap on Spark calls and estimated spend. The student sees a
  friendly message when one is hit.
- **Access code:** set `LENS_ACCESS_CODE`; testers open `/?code=...` once or type it
  on the locked page. The phone never needs it: the pairing QR carries a token that
  works for 15 minutes.
- **Cost log:** each tutor session logs `session cost (estimate)` when it ends: Spark
  tokens and cost (prices from `SPARK_PRICE_*`, default 1.25 / 4.25 USD per M, an
  estimate) plus the GPU seconds it kept warm, and USD per hour.
- **When the shared Spark endpoint stalls** (image reads sometimes queue for a minute
  or more), Lens says "This is taking longer than usual…" (and says it aloud now and
  then), sends a second copy of a slow image read, and gives up on a check after
  170 s; the next step is read as usual. Nothing switches to another model by itself.

### Switching to self-hosted Muse Glimmer by hand (for a demo)

```sh
# 1. In .env: GLIMMER_API_KEY=<any long random string>
make deploy-glimmer        # downloads the weights once (CPU), then deploys muna-lens-glimmer
# 2. In .env, point the brain at it:
#    SPARK_API_KEY=<same value as GLIMMER_API_KEY>
#    SPARK_BASE_URL=https://<workspace>--muna-lens-glimmer-serve.us-east.modal.run/v1
#    SPARK_MODEL=muse-glimmer
#    SPARK_PRICE_IN_PER_M=0 and SPARK_PRICE_OUT_PER_M=0   (paid per GPU-second instead)
# 3. Redeploy the app (make deploy-web) or restart make dev.
# 4. Wake it before the demo: the first request after a cold start waits ~4 min.
curl -H "Authorization: Bearer $GLIMMER_API_KEY" https://<workspace>--muna-lens-glimmer-serve.us-east.modal.run/v1/models
```

While Glimmer is the brain, sessions keep it warm the same way as the other GPU apps
(only while a student is active); it scales to zero 180 s after the last request
(an L40S, about $1.95/h while up). To go back to Spark, remove the three `SPARK_*`
lines (or `SPARK_API_KEY` alone) and redeploy; stop Glimmer with
`modal app stop muna-lens-glimmer` if you want it gone at once.

## Test

```sh
make test        # backend (pytest) + frontend (vitest); offline except one live Spark test when BASETEN is set
make eval        # A5 scorecard per source; needs BASETEN (FERMAT also needs HF_TOKEN)
make test-modal  # the backend tests in a Modal container instead of this machine
make eval-modal ARGS="--source fermat --limit 40"   # the scorecard on Modal
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
  lines.py        A2  handwritten lines from the ink; step boxes snapped onto them
  mathparse.py    A3  one line of LaTeX -> SymPy, or a refusal
  checker.py      A3  does each line follow? first verified error
  judge.py        A3  whole-page model judge (+ confirmation) for lines SymPy cannot parse
  pipeline.py         read -> check -> re-read to confirm -> hint -> outline (shared by API, tutor, eval)
  linecheck.py    F1  a just-finished line alone: two parallel crop reads must agree before a claim
  hints.py        A4  Socratic hint writer + answer-leak guardrail
  pointing.py     C2  fingertip -> step, outlines
  sam.py          C1  client for the SAM still-frame service on Modal (modal_apps/sam.py)
  tracking.py     C1/C2 live SAM 3.1 tracking (modal_apps/sam31.py): relay, writing / pointing / line_done events
  voice.py        V1  client for ASR + TTS on Modal (modal_apps/voice.py)
  tutor.py        F1/V2 tutor session: conversation, spoken hints, barge-in, pointing
  eval/           E0 loaders, A5 harness (`lens-eval`)
  app.py          API, phone pairing + frame relay, serves the built frontend
  access.py           optional access code, phone pairing tokens
  limits.py           rate limits and the daily spend cap
  warm.py             keeps GPU services warm only while a student is active
frontend/src/
  lib/tracking.ts       D1  live tracking client: checks on line_done, outlines follow the page
  lib/changeDetect.ts   D1  new-ink / hand-clear detection (fallback trigger when tracking is off)
  lib/vad.ts, audio.ts  V2  speech detection, mic capture, playback with barge-in
  lib/useTutor.ts       F1  the tutor session in the browser
  lib/rtc.ts                phone -> laptop live video (WebRTC)
  lib/presence.ts           rests Lens when the student is idle or the tab is hidden
  lib/a11y.ts, summary.ts   accessibility settings, end-of-session summary
  modes/                D1  webcam, phone (QR pairing + capture), demo
  components/           D2  Lens screen: page, outline, tutor pointer, steps panel
eval/own/               own eval set labels (images only with the writer's consent)
```

## License

GNU Affero General Public License v3.0 (AGPL-3.0), see [LICENSE](LICENSE). The models keep their own licences (SAM, Omnilingual ASR, Muse Spark Open, Muse Glimmer, Kokoro); the eval datasets are downloaded at run time and never redistributed.
