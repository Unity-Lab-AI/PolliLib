import test from 'node:test';
import assert from 'node:assert/strict';

import { PolliClient } from '../polliLib/index.js';
import { FakeResponse, SeqFetch } from './helpers.js';

test('random seed range + variety', async () => {
  const seq = new SeqFetch();
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const seeds = Array.from({ length: 20 }, () => c._randomSeed());
  assert.ok(seeds.every(s => s >= 10000 && s <= 99999999));
  assert.ok(seeds.some(s => s < 1_000_000));
  assert.ok(seeds.some(s => s >= 10_000_000));
});

test('generate_text JSON and query params', async () => {
  const seq = new SeqFetch([ new FakeResponse({ text: '{"answer":1}' }) ]);
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const out = await c.generate_text('hi', { asJson: true, referrer: 'app', token: 'tok' });
  assert.deepEqual(out, { answer: 1 });
  const call = seq.calls[0];
  assert.match(call.url, /referrer=app/);
  assert.match(call.url, /token=tok/);
});

test('chat_completion payload + content extraction', async () => {
  const seq = new SeqFetch([ new FakeResponse({ jsonData: { choices: [ { message: { content: 'ok' } } ] } }) ]);
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const resp = await c.chat_completion([ { role: 'user', content: 'hi' } ], { referrer: 'r', token: 't' });
  assert.equal(resp, 'ok');
  const call = seq.calls[0];
  const body = JSON.parse(call.opts.body);
  assert.equal(body.referrer, 'r');
  assert.equal(body.token, 't');
});

test('chat_completion_stream SSE yields content chunks', async () => {
  const lines = [
    'event: message',
    'data: {"choices":[{"delta":{"content":"Hel"}}]}',
    'data: {"choices":[{"delta":{"content":"lo"}}]}',
    'data: [DONE]'
  ];
  const seq = new SeqFetch([ new FakeResponse({ streamLines: lines }) ]);
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  let s = '';
  for await (const part of c.chat_completion_stream([ { role:'user', content: 'hi' } ])) s += part;
  assert.equal(s, 'Hello');
});

test('chat_completion_tools two-step', async () => {
  const first = new FakeResponse({ jsonData: { choices: [ { message: { tool_calls: [ { id: 'tc1', function: { name: 'get_current_weather', arguments: JSON.stringify({ location: 'Tokyo', unit: 'celsius' }) } } ] } } ] } });
  const second = new FakeResponse({ jsonData: { choices: [ { message: { content: 'Weather is Cloudy' } } ] } });
  const seq = new SeqFetch([ first, second ]);
  const c = new PolliClient({ fetch: seq.fetch.bind(seq) });
  const out = await c.chat_completion_tools([ { role: 'user', content: 'Weather?' } ], { tools: [ { type: 'function', function: { name: 'get_current_weather', parameters: { type:'object' } } } ], functions: { get_current_weather: ({location, unit}) => ({ location, unit, temperature: '15', description: 'Cloudy' }) } });
  assert.equal(out, 'Weather is Cloudy');
});

