export const FeedsMixin = (Base) => class extends Base {
  async *image_feed_stream({ referrer = null, token = null, timeoutMs = 300_000, reconnect = false, retryDelayMs = 10_000, yieldRawEvents = false, includeBytes = false, includeDataUrl = false } = {}) {
    const feedUrl = new URL('https://image.pollinations.ai/feed');
    if (referrer) feedUrl.searchParams.set('referrer', referrer);
    if (token) feedUrl.searchParams.set('token', token);

    const connect = async function* (self) {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), timeoutMs || self.timeoutMs);
      try {
        const resp = await self.fetch(feedUrl, { method: 'GET', headers: { 'Accept': 'text/event-stream' }, signal: controller.signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        for await (const line of iterateSSELines(resp)) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') break;
          if (yieldRawEvents) { yield data; continue; }
          try {
            const ev = JSON.parse(data);
            if (includeBytes || includeDataUrl) {
              const imgUrl = ev.imageURL || ev.image_url;
              if (imgUrl) {
                const r = await self.fetch(imgUrl);
                if (r.ok) {
                  const buf = Buffer.from(await r.arrayBuffer());
                  if (includeDataUrl) {
                    const ctype = r.headers?.get ? (r.headers.get('Content-Type') || 'image/jpeg') : (r.headers?.['Content-Type'] || 'image/jpeg');
                    const b64 = buf.toString('base64');
                    ev.image_data_url = `data:${ctype};base64,${b64}`;
                  } else if (includeBytes) {
                    ev.image_bytes = buf;
                  }
                }
              }
            }
            yield ev;
          } catch { /* ignore malformed */ }
        }
      } finally { clearTimeout(t); }
    };

    if (!reconnect) {
      yield* (await connect(this));
      return;
    }
    for (;;) {
      try {
        yield* (await connect(this));
      } catch { /* ignore */ }
      await new Promise((r) => setTimeout(r, retryDelayMs));
    }
  }

  async *text_feed_stream({ referrer = null, token = null, timeoutMs = 300_000, reconnect = false, retryDelayMs = 10_000, yieldRawEvents = false } = {}) {
    const feedUrl = new URL('https://text.pollinations.ai/feed');
    if (referrer) feedUrl.searchParams.set('referrer', referrer);
    if (token) feedUrl.searchParams.set('token', token);
    const connect = async function* (self) {
      const controller = new AbortController();
      const t = setTimeout(() => controller.abort(), timeoutMs || self.timeoutMs);
      try {
        const resp = await self.fetch(feedUrl, { method: 'GET', headers: { 'Accept': 'text/event-stream' }, signal: controller.signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        for await (const line of iterateSSELines(resp)) {
          if (!line.startsWith('data:')) continue;
          const data = line.slice(5).trim();
          if (data === '[DONE]') break;
          if (yieldRawEvents) { yield data; continue; }
          try { yield JSON.parse(data); } catch { }
        }
      } finally { clearTimeout(t); }
    };
    if (!reconnect) { yield* (await connect(this)); return; }
    for (;;) {
      try { yield* (await connect(this)); } catch {}
      await new Promise((r) => setTimeout(r, retryDelayMs));
    }
  }
};

async function *iterateSSELines(resp) {
  if (resp.body && typeof resp.body.getReader === 'function') {
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const parts = buf.split(/\r?\n/);
      buf = parts.pop() ?? '';
      for (const l of parts) yield l;
    }
    if (buf) yield buf;
    return;
  }
  const text = await resp.text();
  for (const line of String(text).split(/\r?\n/)) yield line;
}

