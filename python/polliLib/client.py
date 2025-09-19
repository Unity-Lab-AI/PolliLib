from __future__ import annotations

from .base import BaseClient
from .images import ImageMixin
from .text import TextMixin
from .chat import ChatMixin
from .stt import STTMixin
from .vision import VisionMixin
from .feeds import FeedsMixin


class PolliClient(BaseClient, ImageMixin, TextMixin, ChatMixin, STTMixin, VisionMixin, FeedsMixin):
    pass

