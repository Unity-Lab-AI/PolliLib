import { BaseClient } from './base.js';
import { ImagesMixin } from './images.js';
import { TextMixin } from './text.js';
import { ChatMixin } from './chat.js';
import { STTMixin } from './stt.js';
import { VisionMixin } from './vision.js';
import { FeedsMixin } from './feeds.js';

export class PolliClient extends FeedsMixin(VisionMixin(STTMixin(ChatMixin(TextMixin(ImagesMixin(BaseClient)))))) {}

