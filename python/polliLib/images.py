from __future__ import annotations

from typing import Any, Dict, Optional


class ImageMixin:
    def generate_image(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        model: str = "flux",
        seed: Optional[int] = None,
        nologo: bool = True,
        image: Optional[str] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        out_path: Optional[str] = None,
        chunk_size: int = 1024 * 64,
    ) -> bytes | str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive integers")
        if seed is None:
            seed = self._random_seed()
        params: Dict[str, Any] = {
            "width": width,
            "height": height,
            "seed": seed,
            "model": model,
            "nologo": "true" if nologo else "false",
        }
        if image:
            params["image"] = image
        if referrer:
            params["referrer"] = referrer
        if token:
            params["token"] = token

        url = self._image_prompt_url(prompt)
        eff_timeout = timeout if timeout is not None else max(self.timeout, 60.0)
        if out_path:
            with self.session.get(url, params=params, timeout=eff_timeout, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            return out_path
        resp = self.session.get(url, params=params, timeout=eff_timeout)
        resp.raise_for_status()
        return resp.content

    def save_image_timestamped(
        self,
        prompt: str,
        *,
        width: int = 512,
        height: int = 512,
        model: str = "flux",
        nologo: bool = True,
        image: Optional[str] = None,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 300.0,
        images_dir: Optional[str] = None,
        filename_prefix: str = "",
        filename_suffix: str = "",
        ext: str = "jpeg",
    ) -> str:
        import os
        import datetime as dt
        if images_dir is None:
            images_dir = os.path.join(os.getcwd(), "images")
        os.makedirs(images_dir, exist_ok=True)
        ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_ext = (ext or "jpeg").lstrip(".")
        fname = f"{filename_prefix}{ts}{filename_suffix}.{safe_ext}"
        out_path = os.path.join(images_dir, fname)
        return self.generate_image(
            prompt,
            width=width,
            height=height,
            model=model,
            seed=None,
            nologo=nologo,
            image=image,
            referrer=referrer,
            token=token,
            timeout=timeout,
            out_path=out_path,
        )

    def fetch_image(
        self,
        image_url: str,
        *,
        referrer: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = 120.0,
        out_path: Optional[str] = None,
        chunk_size: int = 1024 * 64,
    ) -> bytes | str:
        params: Dict[str, Any] = {}
        if referrer:
            params["referrer"] = referrer
        if token:
            params["token"] = token
        if out_path:
            with self.session.get(image_url, params=params, timeout=timeout or self.timeout, stream=True) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
            return out_path
        resp = self.session.get(image_url, params=params, timeout=timeout or self.timeout)
        resp.raise_for_status()
        return resp.content

