// Single import surface + simple facades
import { PolliClient as _PolliClient } from './client.js';

export const __version__ = '1.0.1';
export class PolliClient extends _PolliClient {}

let _defaultClient = null;
function client() { if (!_defaultClient) _defaultClient = new PolliClient(); return _defaultClient; }

export async function list_models(kind) { return client().listModels(kind); }
export async function get_model_by_name(name, { kind = null } = {}) { return client().getModelByName(name, { kind }); }
export function get_field(model, field, def = null) { return _PolliClient.get(model, field, def); }

// Images
export async function generate_image(prompt, opts) { return client().generate_image(prompt, opts); }
export async function save_image_timestamped(prompt, opts) { return client().save_image_timestamped(prompt, opts); }
export async function fetch_image(imageUrl, opts) { return client().fetch_image(imageUrl, opts); }

// Text
export async function generate_text(prompt, opts) { return client().generate_text(prompt, opts); }

// Chat
export async function chat_completion(messages, opts) { return client().chat_completion(messages, opts); }
export function chat_completion_stream(messages, opts) { return client().chat_completion_stream(messages, opts); }
export async function chat_completion_tools(messages, opts) { return client().chat_completion_tools(messages, opts); }

// STT
export async function transcribe_audio(audioPath, opts) { return client().transcribe_audio(audioPath, opts); }

// Vision
export async function analyze_image_url(imageUrl, opts) { return client().analyze_image_url(imageUrl, opts); }
export async function analyze_image_file(imagePath, opts) { return client().analyze_image_file(imagePath, opts); }

// Feeds
export function image_feed_stream(opts) { return client().image_feed_stream(opts); }
export function text_feed_stream(opts) { return client().text_feed_stream(opts); }
