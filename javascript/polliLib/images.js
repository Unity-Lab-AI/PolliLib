import fs from 'node:fs';
import path from 'node:path';

export const ImagesMixin = (Base) => class extends Base {
  async generate_image(prompt, {
    width = 512,
    height = 512,
    model = 'flux',
    seed = null,
    nologo = true,
    image = null,
    referrer = null,
    token = null,
    timeoutMs = 300_000,
    outPath = null,
    chunkSize = 64 * 1024,
  } = {}) {
    if (!prompt || !String(prompt).trim()) throw new Error('prompt must be a non-empty string');
    width = Number(width); height = Number(height);
    if (!(width > 0) || !(height > 0)) throw new Error('width and height must be positive integers');
    if (seed == null) seed = this._randomSeed();
    const params = new URLSearchParams({ width: String(width), height: String(height), seed: String(seed), model: String(model), nologo: nologo ? 'true' : 'false' });
    if (image) params.set('image', image);
    if (referrer) params.set('referrer', referrer);
    if (token) params.set('token', token);
    const url = this._imagePromptUrl(String(prompt));
    const full = `${url}?${params}`;
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(full, { method: 'GET', signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (outPath) {
        await streamToFile(resp, outPath, chunkSize);
        return outPath;
      }
      const buf = await resp.arrayBuffer();
      return Buffer.from(buf);
    } finally {
      clearTimeout(t);
    }
  }

  async save_image_timestamped(prompt, {
    width = 512,
    height = 512,
    model = 'flux',
    nologo = true,
    image = null,
    referrer = null,
    token = null,
    timeoutMs = 300_000,
    imagesDir = null,
    filenamePrefix = '',
    filenameSuffix = '',
    ext = 'jpeg',
  } = {}) {
    imagesDir ||= path.join(process.cwd(), 'images');
    await fs.promises.mkdir(imagesDir, { recursive: true });
    const ts = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 15); // YYYYMMDDHHMMSS
    const safeExt = String(ext || 'jpeg').replace(/^\./, '');
    const fname = `${filenamePrefix}${ts}${filenameSuffix}.${safeExt}`;
    const outPath = path.join(imagesDir, fname);
    await this.generate_image(prompt, { width, height, model, seed: null, nologo, image, referrer, token, timeoutMs, outPath });
    return outPath;
  }

  async fetch_image(imageUrl, { referrer = null, token = null, timeoutMs = 120_000, outPath = null, chunkSize = 64 * 1024 } = {}) {
    const u = new URL(imageUrl);
    if (referrer) u.searchParams.set('referrer', referrer);
    if (token) u.searchParams.set('token', token);
    const controller = new AbortController();
    const t = setTimeout(() => controller.abort(), timeoutMs || this.timeoutMs);
    try {
      const resp = await this.fetch(u, { method: 'GET', signal: controller.signal });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      if (outPath) {
        await streamToFile(resp, outPath, chunkSize);
        return outPath;
      }
      const buf = await resp.arrayBuffer();
      return Buffer.from(buf);
    } finally {
      clearTimeout(t);
    }
  }
};

async function streamToFile(resp, outPath, chunkSize) {
  const out = fs.createWriteStream(outPath);
  // If web stream body exists
  if (resp.body && typeof resp.body.getReader === 'function') {
    const reader = resp.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) out.write(Buffer.from(value));
    }
    out.end();
    await finished(out);
    return;
  }
  const buf = await resp.arrayBuffer();
  await fs.promises.writeFile(outPath, Buffer.from(buf));
}

function finished(stream) {
  return new Promise((resolve, reject) => {
    stream.on('finish', resolve);
    stream.on('error', reject);
  });
}

