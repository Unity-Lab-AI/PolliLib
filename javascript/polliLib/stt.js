import fs from 'node:fs';

export const STTMixin = (Base) => class extends Base {
  async transcribe_audio(audioPath, { question = 'Transcribe this audio', model = 'openai-audio', provider = 'openai', referrer = null, token = null, timeoutMs = 120_000 } = {}) {
    if (!fs.existsSync(audioPath)) throw new Error(`File not found: ${audioPath}`);
    const ext = String(audioPath).split('.').pop().toLowerCase();
    if (!['mp3','wav'].includes(ext)) return null;
    const data = await fs.promises.readFile(audioPath);
    const b64 = data.toString('base64');
    const payload = {
      model,
      messages: [
        { role: 'user', content: [ { type: 'text', text: question }, { type: 'input_audio', input_audio: { data: b64, format: ext } } ] }
      ]
    };
    if (referrer) payload.referrer = referrer;
    if (token) payload.token = token;
    const url = `${this.textPromptBase}/${provider}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const json = await resp.json();
      return json?.choices?.[0]?.message?.content;
    } finally { clearTimeout(t); }
  }
};

