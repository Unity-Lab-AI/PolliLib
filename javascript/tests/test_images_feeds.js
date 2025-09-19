import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

import { PolliClient } from '../polliLib/index.js';
import { FakeResponse, SeqFetch } from './helpers.js';

test('generate_image streams to file and fetch_image returns bytes', async (t) => {
  const tmpDir = await fs.promises.mkdtemp(path.join(process.cwd(), 'tmp-img-'));
  const outPath = path.join(tmpDir, 'gen.jpg');
  const seq = new SeqFetch([
    new FakeResponse({ content: Buffer.from('abc') }),
  ]);
  // For streaming path: provide body stream lines as bytes
  seq.fetch = async (url, opts = {}) => {
    if (opts?.signal && opts.method === 'GET' && String(url).includes('/prompt/')) {
      // Simulate streaming chunks
      const lines = ['a', 'b', 'c'];
      return new FakeResponse({ streamLines: lines });
    }
    return new FakeResponse({ content: Buffer.from('XYZ') });
  };
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const saved = await c.generate_image('test', { outPath });
  assert.equal(saved, outPath);
  const data = await fs.promises.readFile(outPath);
  // Our fake stream writes chunks with newlines per line; normalize
  assert.equal(data.toString('utf-8').replace(/\n/g, ''), 'abc');

  const bytes = await c.fetch_image('http://x/y.jpg');
  assert.equal(Buffer.isBuffer(bytes), true);
});

test('save_image_timestamped creates file', async () => {
  const tmpDir = await fs.promises.mkdtemp(path.join(process.cwd(), 'tmp-img-'));
  const seq = new SeqFetch([ new FakeResponse({ content: Buffer.from('JPG') }) ]);
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const p = await c.save_image_timestamped('p', { imagesDir: tmpDir });
  assert.ok(fs.existsSync(p));
  assert.ok(p.endsWith('.jpeg'));
});

test('image_feed_stream parses and can include bytes/data URL', async () => {
  const lines = [
    'data: {"prompt":"p1","imageURL":"http://img/1.jpg","model":"flux","seed":123}',
    'data: [DONE]'
  ];
  // Case 1: parse only
  let seq = new SeqFetch([ new FakeResponse({ streamLines: lines }) ]);
  let c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const ev1 = [];
  for await (const ev of c.image_feed_stream()) ev1.push(ev);
  assert.equal(ev1[0].prompt, 'p1');

  // Case 2: include bytes
  seq = new SeqFetch();
  seq.fetch = async (url, opts = {}) => {
    if (String(url).endsWith('/feed')) return new FakeResponse({ streamLines: lines });
    return new FakeResponse({ content: Buffer.from([0xFF,0xD8,0xFF]), headers: { 'Content-Type': 'image/jpeg' } });
  };
  c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const ev2 = [];
  for await (const ev of c.image_feed_stream({ includeBytes: true })) ev2.push(ev);
  assert.equal(Buffer.compare(ev2[0].image_bytes, Buffer.from([0xFF,0xD8,0xFF])), 0);

  // Case 3: include data URL
  seq = new SeqFetch();
  seq.fetch = async (url, opts = {}) => {
    if (String(url).endsWith('/feed')) return new FakeResponse({ streamLines: lines });
    return new FakeResponse({ content: Buffer.from([0xFF,0xD8,0xFF]), headers: { 'Content-Type': 'image/jpeg' } });
  };
  c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const ev3 = [];
  for await (const ev of c.image_feed_stream({ includeDataUrl: true })) ev3.push(ev);
  assert.ok(ev3[0].image_data_url.startsWith('data:image/jpeg;base64,'));
});

test('text_feed_stream parses events', async () => {
  const lines = [
    'data: {"model":"openai","messages":[{"role":"user","content":"hi"}],"response":"Hello"}',
    'data: [DONE]'
  ];
  const seq = new SeqFetch([ new FakeResponse({ streamLines: lines }) ]);
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const it = c.text_feed_stream();
  const first = await (async () => { for await (const ev of it) return ev; })();
  assert.equal(first.model, 'openai');
  assert.equal(first.response, 'Hello');
});
