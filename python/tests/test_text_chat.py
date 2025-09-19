import json
import tempfile
from typing import Dict

from polliLib import PolliClient
from .conftest import FakeResponse, FakeSession


def test_random_seed_range():
    c = PolliClient(session=FakeSession())
    seeds = [c._random_seed() for _ in range(20)]
    assert all(10000 <= s <= 99999999 for s in seeds)
    assert any(s < 10**6 for s in seeds) and any(s >= 10**7 for s in seeds)


def test_generate_text_as_json_and_params():
    fs = FakeSession()
    c = PolliClient(session=fs)

    def fake_get(url, **kw):
        fs.last_get = (url, kw)
        return FakeResponse(text='{"answer": 42}', status=200)

    fs.get = fake_get
    out = c.generate_text("hello", as_json=True, referrer="app", token="tok")
    assert out == {"answer": 42}
    url, kw = fs.last_get
    assert kw["params"]["referrer"] == "app"
    assert kw["params"]["token"] == "tok"


def test_chat_completion_payload_and_extract():
    fs = FakeSession()
    c = PolliClient(session=fs)
    resp = c.chat_completion([{"role": "user", "content": "hi"}], referrer="r", token="t")
    assert resp == "ok"
    url, headers, payload, kw = fs.last_post
    assert payload["referrer"] == "r" and payload["token"] == "t"


def test_chat_completion_stream_sse():
    lines = [
        'event: message',
        'data: {"choices":[{"delta":{"content":"Hel"}}]}',
        'data: {"choices":[{"delta":{"content":"lo"}}]}',
        'data: [DONE]'
    ]
    fs = FakeSession()
    fs.post = lambda url, **kw: FakeResponse(stream_lines=lines)
    c = PolliClient(session=fs)
    chunks = list(c.chat_completion_stream([{"role": "user", "content": "x"}]))
    assert "".join(chunks) == "Hello"


def test_chat_completion_tools_one_round():
    # First call returns tool_call; second returns final message
    first_data: Dict = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "tc1",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": json.dumps({"location": "Tokyo", "unit": "celsius"}),
                            },
                        }
                    ]
                }
            }
        ]
    }
    second_data: Dict = {
        "choices": [
            {
                "message": {
                    "content": "Weather is Cloudy"
                }
            }
        ]
    }

    class SeqSession(FakeSession):
        def __init__(self):
            super().__init__()
            self.count = 0

        def post(self, url, headers=None, json=None, **kw):
            self.count += 1
            self.last_post = (url, headers or {}, json or {}, kw)
            return FakeResponse(json_data=(first_data if self.count == 1 else second_data))

    fs = SeqSession()
    c = PolliClient(session=fs)

    def get_current_weather(location: str, unit: str = "celsius"):
        return {"location": location, "temperature": "15", "unit": unit, "description": "Cloudy"}

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "mock",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}},
            },
        }
    ]
    msg = [{"role": "user", "content": "What's weather?"}]
    out = c.chat_completion_tools(msg, tools=tools, functions={"get_current_weather": get_current_weather})
    assert out == "Weather is Cloudy"

