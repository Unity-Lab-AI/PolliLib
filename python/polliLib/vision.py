from __future__ import annotations

from typing import Any, Dict, Optional


class VisionMixin:
    def analyze_image_url(
        self,
        image_url: str,
        *,
        question: str = "What's in this image?",
        model: str = "openai",
        max_tokens: Optional[int] = 500,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        as_json: bool = False,
    ) -> Any:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token
        url = f"{self.text_prompt_base}/{model}"
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if as_json:
            return data
        return data.get("choices", [{}])[0].get("message", {}).get("content")

    def analyze_image_file(
        self,
        image_path: str,
        *,
        question: str = "What's in this image?",
        model: str = "openai",
        max_tokens: Optional[int] = 500,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 60.0,
        as_json: bool = False,
    ) -> Any:
        import os, base64
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        if ext not in {"jpeg", "jpg", "png", "gif", "webp"}:
            ext = "jpeg"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:image/{ext};base64,{b64}"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token
        url = f"{self.text_prompt_base}/{model}"
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if as_json:
            return data
        return data.get("choices", [{}])[0].get("message", {}).get("content")

