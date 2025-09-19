export const TextMixin = (Base) => class extends Base {
  async generate_text(prompt, { model = 'openai', seed = null, system = null, referrer = null, token = null, asJson = false, timeoutMs = 60_000 } = {}) {
    if (!prompt || !String(prompt).trim()) throw new Error('prompt must be a non-empty string');
    if (seed == null) seed = this._randomSeed();
    const url = new URL(this._textPromptUrl(String(prompt)));
    url.searchParams.set('model', model);
    url.searchParams.set('seed', String(seed));
    if (asJson) url.searchParams.set('json', 'true');
    if (system) url.searchParams.set('system', system);
    if (referrer) url.searchParams.set('referrer', referrer);
    if (token) url.searchParams.set('token', token);
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(url, { method: 'GET', signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (asJson) {
        const text = await resp.text();
        try { return JSON.parse(text); } catch { return text; }
      }
      return await resp.text();
    } finally {
      clearTimeout(t);
    }
  }
};
