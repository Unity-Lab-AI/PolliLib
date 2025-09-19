// Base utilities and model listing

const DEFAULTS = {
  textUrl: "https://text.pollinations.ai/models",
  imageUrl: "https://image.pollinations.ai/models",
  imagePromptBase: "https://image.pollinations.ai/prompt",
  textPromptBase: "https://text.pollinations.ai",
  timeoutMs: 10000,
};

export class BaseClient {
  constructor(opts = {}) {
    this.textUrl = opts.textUrl || DEFAULTS.textUrl;
    this.imageUrl = opts.imageUrl || DEFAULTS.imageUrl;
    this.imagePromptBase = opts.imagePromptBase || DEFAULTS.imagePromptBase;
    this.textPromptBase = opts.textPromptBase || DEFAULTS.textPromptBase;
    this.timeoutMs = opts.timeoutMs || DEFAULTS.timeoutMs;
    this.fetch = opts.fetch || (typeof fetch !== "undefined" ? fetch.bind(globalThis) : null);
    if (!this.fetch) throw new Error("fetch is not available; provide opts.fetch");
    this._modelsCache = new Map(); // kind -> list
  }

  async listModels(kind /* 'text' | 'image' */) {
    const cached = this._modelsCache.get(kind);
    if (cached) return cached;
    const url = this._url(kind);
    const resp = await this.fetch(url, { method: "GET" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const json = await resp.json();
    const models = this._normalizeModels(json);
    this._modelsCache.set(kind, models);
    return models;
  }

  async getModelByName(name, { kind = null, includeAliases = true, caseInsensitive = true } = {}) {
    const needle = caseInsensitive ? String(name).toLowerCase() : String(name);
    const kinds = kind ? [kind] : ["text", "image"];
    for (const k of kinds) {
      const models = await this.listModels(k);
      for (const m of models) {
        let names = [m.name || ""];
        if (includeAliases && Array.isArray(m.aliases)) names = names.concat(m.aliases);
        if (caseInsensitive) names = names.map((n) => String(n).toLowerCase());
        if (names.includes(needle)) return m;
      }
    }
    return null;
  }

  static get(model, field, def = null) {
    return model && Object.prototype.hasOwnProperty.call(model, field) ? model[field] : def;
  }

  refreshCache() {
    this._modelsCache.clear();
  }

  _url(kind) {
    return kind === "text" ? this.textUrl : this.imageUrl;
  }

  _normalizeModels(raw) {
    if (raw && typeof raw === "object" && !Array.isArray(raw) && Array.isArray(raw.models)) {
      raw = raw.models;
    }
    if (!Array.isArray(raw)) return [];
    const out = [];
    for (const item of raw) {
      if (typeof item === "string") {
        out.push({
          name: item,
          aliases: [],
          input_modalities: [],
          output_modalities: [],
          tools: false,
          vision: false,
          audio: false,
          community: false,
          supportsSystemMessages: true,
        });
      } else if (item && typeof item === "object") {
        const m = { ...item };
        if ("teir" in m && !("tier" in m)) m.tier = m.teir, delete m.teir;
        m.aliases ||= [];
        m.input_modalities ||= [];
        m.output_modalities ||= [];
        m.tools ||= false;
        m.vision ||= false;
        m.audio ||= false;
        m.community ||= false;
        m.supportsSystemMessages = m.supportsSystemMessages ?? true;
        out.push(m);
      }
    }
    return out;
  }

  _randomSeed() {
    const digits = Math.floor(Math.random() * 4) + 5; // 5..8
    const low = 10 ** (digits - 1);
    const high = 10 ** digits - 1;
    return Math.floor(Math.random() * (high - low + 1)) + low;
  }

  _imagePromptUrl(prompt) {
    return `${this.imagePromptBase}/${encodeURIComponent(prompt)}`;
  }

  _textPromptUrl(prompt) {
    return `${this.textPromptBase}/${encodeURIComponent(prompt)}`;
  }
}

