// Simple fetch stubs for tests
export class FakeHeaders {
  constructor(init = {}) { this.map = { ...init }; }
  get(name) { return this.map[name] || this.map[name?.toLowerCase()] || null; }
}

export class FakeResponse {
  constructor({ status = 200, text = null, jsonData = null, streamLines = null, content = null, headers = {} } = {}) {
    this.status = status;
    this.ok = status >= 200 && status < 300;
    this._text = text;
    this._json = jsonData;
    this._lines = streamLines; // array of sse lines
    this._content = content; // Buffer
    this.headers = new FakeHeaders(headers);
  }
  async json() { if (this._json != null) return this._json; return JSON.parse(await this.text()); }
  async text() { if (this._text != null) return this._text; if (this._content) return this._content.toString('utf-8'); if (this._lines) return this._lines.join('\n'); return ''; }
  async arrayBuffer() { if (this._content) return this._content; const t = await this.text(); return Buffer.from(t, 'utf-8'); }
  get body() {
    if (!this._lines) return null;
    const encoder = new TextEncoder();
    const lines = this._lines.slice();
    return new ReadableStream({
      pull(controller) {
        if (!lines.length) { controller.close(); return; }
        const chunk = lines.shift() + '\n';
        controller.enqueue(encoder.encode(chunk));
      }
    });
  }
}

export class SeqFetch {
  constructor(responses = []) { this.responses = responses; this.calls = []; }
  async fetch(url, opts = {}) { this.calls.push({ url: String(url), opts }); return this.responses.length ? this.responses.shift() : new FakeResponse(); }
}

