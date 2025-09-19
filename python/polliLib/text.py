from __future__ import annotations

from typing import Any, Dict, Optional


class TextMixin:
    def generate_text(
        self,
        prompt: str,
        *,
        model: str = "openai",
        seed: Optional[int] = None,
        system: Optional[str] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        as_json: bool = False,
        timeout: Optional[float] = 60.0,
    ) -> Any:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if seed is None:
            seed = self._random_seed()
        params: Dict[str, Any] = {
            "model": model,
            "seed": seed,
        }
        if as_json:
            params["json"] = "true"
        if system:
            params["system"] = system
        if referrer:
            params["referrer"] = referrer
        if token:
            params["token"] = token
        url = self._text_prompt_url(prompt)
        eff_timeout = timeout if timeout is not None else max(self.timeout, 10.0)
        resp = self.session.get(url, params=params, timeout=eff_timeout)
        resp.raise_for_status()
        if as_json:
            import json as _json
            txt = resp.text
            try:
                return _json.loads(txt)
            except Exception:
                return txt
        else:
            return resp.text

