import fs from 'node:fs';

export const VisionMixin = (Base) => class extends Base {
  async analyze_image_url(imageUrl, { question = "What's in this image?", model = 'openai', max_tokens = 500, referrer = null, token = null, timeoutMs = 60_000, asJson = false } = {}) {
    const payload = {
      model,
      messages: [ { role: 'user', content: [ { type: 'text', text: question }, { type: 'image_url', image_url: { url: imageUrl } } ] } ],
    };
    if (typeof max_tokens === 'number') payload.max_tokens = max_tokens;
    if (referrer) payload.referrer = referrer;
    if (token) payload.token = token;
    const url = `${this.textPromptBase}/${model}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      if (asJson) return json;
      return json?.choices?.[0]?.message?.content;
    } finally { clearTimeout(t); }
  }

  async analyze_image_file(imagePath, { question = "What's in this image?", model = 'openai', max_tokens = 500, referrer = null, token = null, timeoutMs = 60_000, asJson = false } = {}) {
    if (!fs.existsSync(imagePath)) throw new Error(`File not found: ${imagePath}`);
    let ext = String(imagePath).split('.').pop().toLowerCase();
    if (!['jpeg','jpg','png','gif','webp'].includes(ext)) ext = 'jpeg';
    const data = await fs.promises.readFile(imagePath);
    const b64 = data.toString('base64');
    const dataUrl = `data:image/${ext};base64,${b64}`;
    const payload = {
      model,
      messages: [ { role: 'user', content: [ { type: 'text', text: question }, { type: 'image_url', image_url: { url: dataUrl } } ] } ],
    };
    if (typeof max_tokens === 'number') payload.max_tokens = max_tokens;
    if (referrer) payload.referrer = referrer;
    if (token) payload.token = token;
    const url = `${this.textPromptBase}/${model}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      if (asJson) return json;
      return json?.choices?.[0]?.message?.content;
    } finally { clearTimeout(t); }
  }
};

