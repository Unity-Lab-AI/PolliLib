export const ChatMixin = (Base) => class extends Base {
  async chat_completion(messages, { model = 'openai', seed = null, private_: priv = undefined, referrer = null, token = null, asJson = false, timeoutMs = 60_000 } = {}) {
    if (!Array.isArray(messages) || messages.length === 0) throw new Error('messages must be a non-empty list');
    if (seed == null) seed = this._randomSeed();
    const payload = { model, messages, seed };
    if (priv !== undefined) payload.private = !!priv;
    if (referrer) payload.referrer = referrer;
    if (token) payload.token = token;
    const url = `${this.textPromptBase}/${model}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      if (asJson) return data;
      try { return data?.choices?.[0]?.message?.content; } catch { return JSON.stringify(data); }
    } finally { clearTimeout(t); }
  }

  async *chat_completion_stream(messages, { model = 'openai', seed = null, private_: priv = undefined, referrer = null, token = null, timeoutMs = 300_000, yieldRawEvents = false } = {}) {
    if (!Array.isArray(messages) || messages.length === 0) throw new Error('messages must be a non-empty list');
    if (seed == null) seed = this._randomSeed();
    const payload = { model, messages, seed, stream: true };
    if (priv !== undefined) payload.private = !!priv;
    if (referrer) payload.referrer = referrer;
    if (token) payload.token = token;
    const url = `${this.textPromptBase}/${model}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' }, body: JSON.stringify(payload), signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      for await (const line of iterateSSELines(resp)) {
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') break;
        if (yieldRawEvents) { yield data; continue; }
        try {
          const obj = JSON.parse(data);
          const content = obj?.choices?.[0]?.delta?.content;
          if (content) yield content;
        } catch { /* ignore */ }
      }
    } finally { clearTimeout(t); }
  }

  async chat_completion_tools(messages, { tools, functions = {}, tool_choice = 'auto', model = 'openai', seed = null, private_: priv = undefined, referrer = null, token = null, asJson = false, timeoutMs = 60_000, max_rounds = 1 } = {}) {
    if (!Array.isArray(messages) || messages.length === 0) throw new Error('messages must be a non-empty list');
    if (!Array.isArray(tools) || tools.length === 0) throw new Error('tools must be a non-empty list');
    if (seed == null) seed = this._randomSeed();
    const url = `${this.textPromptBase}/${model}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const history = [...messages];
      let rounds = 0;
      for (;;) {
        const payload = { model, messages: history, seed, tools, tool_choice };
        if (priv !== undefined) payload.private = !!priv;
        if (referrer) payload.referrer = referrer;
        if (token) payload.token = token;
        const resp = await this.fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal: controller.signal });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const msg = data?.choices?.[0]?.message || {};
        const toolCalls = msg?.tool_calls || [];
        if (!toolCalls.length || rounds >= max_rounds) {
          if (asJson) return data;
          return msg?.content;
        }
        history.push(msg);
        for (const tc of toolCalls) {
          const fnName = tc?.function?.name;
          const argsText = tc?.function?.arguments ?? '{}';
          let args = {};
          try { args = typeof argsText === 'string' ? JSON.parse(argsText) : (argsText || {}); } catch { args = {}; }
          let result;
          if (functions && Object.prototype.hasOwnProperty.call(functions, fnName)) {
            try {
              if (Array.isArray(args)) {
                result = await functions[fnName](...args);
              } else if (args && typeof args === 'object') {
                result = await functions[fnName](args);
              } else {
                result = await functions[fnName]();
              }
            } catch (e) { result = { error: `function '${fnName}' raised: ${e}` }; }
          } else {
            result = { error: `no handler for function '${fnName}'` };
          }
          const content = typeof result === 'string' ? result : JSON.stringify(result);
          history.push({ tool_call_id: tc.id, role: 'tool', name: fnName, content });
        }
        rounds += 1;
      }
    } finally { clearTimeout(t); }
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
      for (const line of parts) yield line;
    }
    if (buf) yield buf;
    return;
  }
  const text = await resp.text();
  for (const line of String(text).split(/\r?\n/)) yield line;
}
