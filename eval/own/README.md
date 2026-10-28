# Own eval set

Real university calculus, physics and chemistry pages, shot on a phone and a
webcam (tilt, shadows, a hand in frame), some with a planted mistake. These
pages cover the camera conditions and subjects that FERMAT and EDU-CIRCUIT-HW
lack.

## Consent rule

Commit a page image here **only with its writer's consent**. Otherwise put the
image in `.cache/eval/own/` (gitignored) and still list it in `labels.json`;
the loader looks in both places.

## labels.json

One entry per page:

```json
{
  "file": "calc-01-phone.jpg",
  "problem": "Differentiate x^2 sin x",
  "first_error_step": 2,
  "subject": "calculus",
  "camera": "phone",
  "conditions": "tilted, shadow top-left",
  "transcription": "optional: the lines as LaTeX, one per line"
}
```

- `first_error_step`: 0-based index of the first wrong line, or `"none"`.
- `camera`: `"phone"` or `"webcam"`.

Run the scorecard with `uv run lens-eval --source own` (needs `BASETEN` in `.env`).
