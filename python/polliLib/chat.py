from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Callable


class ChatMixin:
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "openai",
        seed: Optional[int] = None,
        private: Optional[bool] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        as_json: bool = False,
        timeout: Optional[float] = 60.0,
    ) -> Any:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")
        if seed is None:
            seed = self._random_seed()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "seed": seed,
        }
        if private is not None:
            payload["private"] = bool(private)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token
        url = f"{self.text_prompt_base}/{model}"
        eff_timeout = timeout if timeout is not None else max(self.timeout, 10.0)
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=eff_timeout)
        resp.raise_for_status()
        data = resp.json()
        if as_json:
            return data
        try:
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
        except Exception:
            return resp.text

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "openai",
        seed: Optional[int] = None,
        private: Optional[bool] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        yield_raw_events: bool = False,
    ) -> Iterator[str]:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of {role, content} dicts")
        if seed is None:
            seed = self._random_seed()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "seed": seed,
            "stream": True,
        }
        if private is not None:
            payload["private"] = bool(private)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token
        url = f"{self.text_prompt_base}/{model}"
        eff_timeout = timeout if timeout is not None else max(self.timeout, 60.0)
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        with self.session.post(url, headers=headers, json=payload, timeout=eff_timeout, stream=True) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8", errors="ignore")
                    except Exception:
                        continue
                line = raw.strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                if yield_raw_events:
                    yield data
                    continue
                try:
                    import json as _json
                    obj = _json.loads(data)
                    content = (
                        obj.get("choices", [{}])[0]
                        .get("delta", {})
                        .get("content")
                    )
                    if content:
                        yield content
                except Exception:
                    continue

    def chat_completion_tools(
        self,
        messages: List[Dict[str, Any]],
        *,
        tools: List[Dict[str, Any]],
        functions: Optional[Dict[str, Callable[..., Any]]] = None,
        tool_choice: Any = "auto",
        model: str = "openai",
        seed: Optional[int] = None,
        private: Optional[bool] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        as_json: bool = False,
        timeout: Optional[float] = 60.0,
        max_rounds: int = 1,
    ) -> Any:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list of messages")
        if not isinstance(tools, list) or not tools:
            raise ValueError("tools must be a non-empty list of tool specs")
        if seed is None:
            seed = self._random_seed()
        url = f"{self.text_prompt_base}/{model}"
        headers = {"Content-Type": "application/json"}
        eff_timeout = timeout if timeout is not None else max(self.timeout, 10.0)
        history: List[Dict[str, Any]] = list(messages)
        rounds = 0
        while True:
            payload: Dict[str, Any] = {
                "model": model,
                "messages": history,
                "seed": seed,
                "tools": tools,
                "tool_choice": tool_choice,
            }
            if private is not None:
                payload["private"] = bool(private)
            if referrer:
                payload["referrer"] = referrer
            if token:
                payload["token"] = token
            resp = self.session.post(url, headers=headers, json=payload, timeout=eff_timeout)
            resp.raise_for_status()
            data = resp.json()
            msg = (data.get("choices", [{}])[0]).get("message", {})
            tool_calls = msg.get("tool_calls", []) or []
            if not tool_calls or rounds >= max_rounds:
                if as_json:
                    return data
                return msg.get("content")
            history.append(msg)
            for tc in tool_calls:
                fn_name = tc.get("function", {}).get("name")
                args_text = tc.get("function", {}).get("arguments", "{}")
                try:
                    import json as _json
                    args = _json.loads(args_text) if isinstance(args_text, str) else (args_text or {})
                except Exception:
                    args = {}
                if functions and fn_name in functions:
                    try:
                        result = functions[fn_name](**args) if isinstance(args, dict) else functions[fn_name]()
                    except Exception as e:
                        result = {"error": f"function '{fn_name}' raised: {e}"}
                else:
                    result = {"error": f"no handler for function '{fn_name}'"}
                if not isinstance(result, str):
                    import json as _json
                    content_str = _json.dumps(result)
                else:
                    content_str = result
                history.append(
                    {
                        "tool_call_id": tc.get("id"),
                        "role": "tool",
                        "name": fn_name,
                        "content": content_str,
                    }
                )
            rounds += 1

