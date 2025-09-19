from __future__ import annotations

from typing import Any, Dict, Optional


class STTMixin:
    def transcribe_audio(
        self,
        audio_path: str,
        *,
        question: str = "Transcribe this audio",
        model: str = "openai-audio",
        provider: str = "openai",
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 120.0,
    ) -> Optional[str]:
        import os, base64
        if not os.path.exists(audio_path):
            raise FileNotFoundError(audio_path)
        ext = os.path.splitext(audio_path)[1].lower().lstrip(".")
        if ext not in {"mp3", "wav"}:
            return None
        with open(audio_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "input_audio", "input_audio": {"data": b64, "format": ext}},
                    ],
                }
            ],
        }
        if referrer:
            payload["referrer"] = referrer
        if token:
            payload["token"] = token
        url = f"{self.text_prompt_base}/{provider}"
        headers = {"Content-Type": "application/json"}
        resp = self.session.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content")

