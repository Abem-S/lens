"""A1: the client sends the right payloads, retries, streams and logs cost.

Offline tests use a mocked HTTP transport. The live test runs only when BASETEN
is set (it is the A1 done criterion: one real call of each input type).
"""

import json
import os

import httpx
import pytest

from lens.spark import EmptyReply, SparkClient, audio_part, image_part, text_part


def completion(text, p=10, c=20):
    return {
        "id": "x", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
    }


def client_with(handler, **kw):
    return SparkClient(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)), **kw)


def test_text_call_sends_effort_and_logs_cost():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=completion("hello", 1000, 2000))

    c = client_with(handler, price_in=1.0, price_out=2.0)
    r = c.ask("hi", effort="minimal")
    assert r.text == "hello"
    assert seen["reasoning_effort"] == "minimal"
    assert seen["model"] == "meta-models/muse-spark-open-1026"
    assert seen["max_tokens"] >= 1500
    assert r.log.cost_usd == pytest.approx((1000 * 1 + 2000 * 2) / 1e6)
    assert c.total_cost == pytest.approx(0.005)


def test_image_and_audio_parts_are_data_urls():
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=completion("ok"))

    c = client_with(handler)
    c.ask([text_part("read"), image_part(b"\xff\xd8jpeg"), audio_part(b"RIFFwav")])
    parts = seen["messages"][0]["content"]
    assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert parts[2]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert c.calls[0].cost_usd is None  # no prices configured


def test_retries_on_server_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("lens.spark.time.sleep", lambda s: None)
    attempts = {"n": 0}

    def handler(req):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(200, json=completion("done"))

    assert client_with(handler).ask("hi").text == "done"
    assert attempts["n"] == 3


def test_empty_reply_raises():
    with pytest.raises(EmptyReply):
        client_with(lambda req: httpx.Response(200, json=completion(None))).ask("hi")


def test_streaming_yields_deltas_and_logs_usage():
    def sse(obj):
        return f"data: {json.dumps(obj)}\n\n"

    def chunk(text=None, usage=None):
        return {"id": "x", "object": "chat.completion.chunk", "created": 0, "model": "m",
                "choices": [] if text is None else [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                "usage": usage}

    body = sse(chunk("Wh")) + sse(chunk("at?")) + sse(chunk(usage={"prompt_tokens": 5, "completion_tokens": 7,
                                                                   "total_tokens": 12})) + "data: [DONE]\n\n"

    def handler(req):
        assert json.loads(req.content)["stream"] is True
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    c = client_with(handler)
    assert "".join(c.stream([{"role": "user", "content": "hi"}], effort="minimal")) == "What?"
    assert (c.calls[0].prompt_tokens, c.calls[0].completion_tokens) == (5, 7)


@pytest.mark.skipif(not os.environ.get("BASETEN"), reason="needs BASETEN key (live A1 check)")
def test_live_each_input_type():  # pragma: no cover - live
    import io
    import math
    import struct
    import wave

    from PIL import Image, ImageDraw

    from lens.config import Settings

    c = SparkClient.from_settings(Settings.from_env())
    assert c.ask("Reply with the word ok.", effort="minimal").text

    img = Image.new("RGB", (400, 120), "white")
    ImageDraw.Draw(img).text((20, 40), "2x + 5 = 11", fill="black")
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    assert c.ask([text_part("What is written?"), image_part(buf.getvalue())], effort="low").text

    wav = io.BytesIO()
    with wave.open(wav, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000)
        w.writeframes(b"".join(struct.pack("<h", int(3000 * math.sin(i / 8))) for i in range(16000)))
    assert c.ask([text_part("Describe this audio."), audio_part(wav.getvalue())], effort="low").text
    assert all(call.prompt_tokens > 0 for call in c.calls)
