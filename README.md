# Muna Lens

A tutor that watches a student solve STEM problems on paper. A camera sees the
page; Lens reads each handwritten line, finds the **first real mistake**, outlines
it, and asks one Socratic question. It never gives the answer.

Intelligence runs on Meta open
models (Muse Spark Open, SAM 3.1, Omnilingual ASR); see [AGENTS.md](AGENTS.md) for
the plan, rules and build order.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

```sh
cp .env.example .env      # add BASETEN to enable page checks
make install
make dev                  # open http://localhost:5173
```

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
make test        # backend (pytest) + frontend (vitest), all offline
make eval        # A5 scorecard per source; needs BASETEN (FERMAT also needs HF_TOKEN)
```

## Layout

```
backend/src/lens/
  contract.py     Page / Step / Finding (the shared contract)
  spark.py        A1  Muse Spark Open client (text, image, audio, streaming, cost log)
  reader.py       A2  frame -> Page
  mathparse.py    A3  one line of LaTeX -> SymPy, or a refusal
  checker.py      A3  does each line follow? first verified error
  judge.py        A3  model judge for lines SymPy cannot parse
  hints.py        A4  Socratic hint writer + answer-leak guardrail
  pointing.py     C2  fingertip -> step, outlines
  eval/           E0 loaders, A5 harness (`lens-eval`)
  app.py          API, phone pairing + frame relay, serves the built frontend
frontend/src/
  lib/changeDetect.ts   D1  new-ink / hand-clear detection (decides when to call Spark)
  modes/                D1  webcam, phone (QR pairing + capture), demo
  components/           D2  Lens screen: page, outline, tutor pointer, steps panel
eval/own/               own eval set labels (images only with the writer's consent)
```
