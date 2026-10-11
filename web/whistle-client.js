// The page's side of the worker (worker.js): calls as promises, the model
// download with its cache and SHA-256 check, audio decoding and the stream
// session. It works in insecure contexts (a page over plain http other
// than localhost) too, which have no Web Crypto and no Cache Storage. No
// DOM here, so the headless test (cmd/whistle-wasm/node) runs the same
// code.

// The pinned model, as `whistle download` fetches it (internal/hub/manifest.lock.json).
export const PINNED_MODEL = {
  url: "https://huggingface.co/Cactus-Compute/whistle/resolve/b358ddadd89b7a713b5aa131f23032d3cca1b251/whistle.cact",
  sha256: "b6e02f048568ac5d01a2042556c658061e699acbc0aa2a1439f52f3d461dffeb",
  size: 16919407,
};

export const SAMPLE_RATE = 16000;

// MAX_SAMPLES is whistle.MaxSamples, the most one call takes (30 s).
export const MAX_SAMPLES = 30 * SAMPLE_RATE;

const CACHE_NAME = "whistle-models-v1";

// WhistleClient talks to a worker running worker.js.
export class WhistleClient {
  constructor(worker) {
    this.worker = worker;
    this.next = 1;
    this.pending = new Map();
    this.ready = new Promise((resolve, reject) => {
      this.readyResolve = resolve;
      this.readyReject = reject;
    });

    worker.onmessage = (event) => this.receive(event.data);
    worker.onerror = (event) =>
      this.fail(new Error(event.message || "worker error"));
  }

  receive(msg) {
    if (msg.type === "ready") {
      this.readyResolve(msg.version);
      return;
    }

    if (msg.type === "fatal") {
      this.fail(new Error(msg.error));
      return;
    }

    const call = this.pending.get(msg.id);
    if (!call) {
      return;
    }

    this.pending.delete(msg.id);
    if ("error" in msg) {
      call.reject(new Error(msg.error));
    } else {
      call.resolve(msg.result);
    }
  }

  fail(err) {
    this.dead = err;
    this.readyReject(err);
    for (const call of this.pending.values()) {
      call.reject(err);
    }
    this.pending.clear();
  }

  call(op, args = [], transfer = []) {
    if (this.dead) {
      return Promise.reject(this.dead);
    }
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, op, args }, transfer);
    });
  }

  // load hands the model bytes (an ArrayBuffer, given away) to the worker
  // and resolves to {languages, bytes}.
  load(buffer) {
    return this.call("load", [buffer], [buffer]);
  }

  // sha256 hands buffer (an ArrayBuffer) to the worker, which hashes it with
  // Go's crypto/sha256 and gives it back rather than a copy: it resolves to
  // {sha256, buffer}, and the caller goes on with that buffer.
  sha256(buffer) {
    return this.call("sha256", [buffer], [buffer]);
  }

  // decodeWAV reads a WAV file (an ArrayBuffer, given away) as `whistle
  // transcribe` does and resolves to 16 kHz mono samples.
  decodeWAV(buffer) {
    return this.call("decodeWAV", [buffer], [buffer]);
  }

  // transcribe resolves to the engine's JSON, parsed; json holds the string.
  async transcribe(pcm, opts = {}) {
    const json = await this.call("transcribe", [pcm, opts]);
    return { ...JSON.parse(json), json };
  }

  streamStart(opts = {}) {
    return this.call("streamStart", [opts]);
  }

  async streamProcess(pcm) {
    const json = await this.call("streamProcess", [pcm]);
    return { ...JSON.parse(json), json };
  }

  async streamStop() {
    const json = await this.call("streamStop");
    return { ...JSON.parse(json), json };
  }
}

// sha256Of hashes buffer (an ArrayBuffer) with Web Crypto, or, where there
// is none, with Go in client's worker: crypto.subtle exists only in secure
// contexts (https, localhost). It resolves to {sha256, buffer, hashedIn},
// hashedIn "Web Crypto" or "Go"; buffer is the one to go on with, since
// the worker takes the bytes and gives them back rather than copying them.
export async function sha256Of(buffer, client) {
  if (globalThis.crypto?.subtle) {
    const digest = await globalThis.crypto.subtle.digest("SHA-256", buffer);
    const sha256 = Array.from(new Uint8Array(digest), (b) =>
      b.toString(16).padStart(2, "0"),
    ).join("");
    return { sha256, buffer, hashedIn: "Web Crypto" };
  }

  if (!client) {
    throw new Error(
      "no SHA-256: Web Crypto needs a secure context (https or localhost), and no worker was given",
    );
  }
  const got = await client.sha256(buffer);
  return { ...got, hashedIn: "Go" };
}

// fetchModel returns the bytes behind url from the Cache Storage, or
// downloads and caches them. With sha256 set, the bytes must hash to it,
// cached or not: a cached copy that does not is dropped and fetched again,
// and a download that does not is rejected and not cached. onProgress gets
// (loaded, total) bytes during a download. The cache is optional: one that
// is missing (an insecure context) or cannot be opened, read or written
// (storage disabled or denied, the quota full) counts as a miss and the
// download goes ahead. client hashes where there is no Web Crypto
// (sha256Of). It resolves to {buffer, sha256, hashedIn, cached}.
export async function fetchModel(
  url,
  { sha256 = "", client = null, onProgress = () => {}, fetchFn = fetch } = {},
) {
  let cache = null;
  try {
    cache = globalThis.caches ? await globalThis.caches.open(CACHE_NAME) : null;
  } catch {
    // no cache
  }

  if (cache) {
    try {
      const hit = await cache.match(url);
      if (hit) {
        const got = await sha256Of(await hit.arrayBuffer(), client);
        if (!sha256 || got.sha256 === sha256) {
          return { ...got, cached: true };
        }
        await cache.delete(url);
      }
    } catch {
      // a miss
    }
  }

  const res = await fetchFn(url);
  if (!res.ok) {
    throw new Error(`${url}: HTTP ${res.status}`);
  }

  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    parts.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }

  const bytes = new Uint8Array(loaded);
  let at = 0;
  for (const p of parts) {
    bytes.set(p, at);
    at += p.length;
  }

  const got = await sha256Of(bytes.buffer, client);
  if (sha256 && got.sha256 !== sha256) {
    throw new Error(`${url}: SHA-256 ${got.sha256}, want ${sha256}`);
  }

  if (cache) {
    try {
      await cache.put(
        url,
        new Response(got.buffer.slice(0), {
          headers: { "content-type": "application/octet-stream" },
        }),
      );
    } catch {
      // not cached
    }
  }

  return { ...got, cached: false };
}

// decodeAudio turns an audio file into 16 kHz mono samples. A WAV of
// integer PCM goes through the CLI's own reader in Go (decoder "go"), so it
// transcribes exactly as `whistle transcribe` does; anything else the
// browser decodes and resamples with an OfflineAudioContext (decoder
// "browser"), which can differ from the CLI's soxr resampler in the last
// bits.
export async function decodeAudio(client, buffer) {
  try {
    const pcm = await client.decodeWAV(buffer.slice(0));
    return { pcm, decoder: "go" };
  } catch (goErr) {
    if (typeof OfflineAudioContext === "undefined") {
      throw goErr;
    }
    try {
      return {
        pcm: await decodeInBrowser(buffer.slice(0)),
        decoder: "browser",
        reason: goErr.message,
      };
    } catch (err) {
      throw new Error(
        `${goErr.message}; the browser cannot decode it either: ${err.message}`,
      );
    }
  }
}

async function decodeInBrowser(buffer) {
  // Any rate will do to decode; the offline context then mixes to mono and
  // resamples to 16 kHz.
  const ctx = new OfflineAudioContext(1, 1, SAMPLE_RATE);
  const audio = await ctx.decodeAudioData(buffer);
  const frames = Math.ceil(audio.duration * SAMPLE_RATE);
  if (frames === 0) {
    throw new Error("no audio");
  }

  const offline = new OfflineAudioContext(1, frames, SAMPLE_RATE);
  const src = offline.createBufferSource();
  src.buffer = audio;
  src.connect(offline.destination);
  src.start();

  const out = await offline.startRendering();
  return out.getChannelData(0).slice();
}

// StreamSession runs one stream: feed appends samples, and one
// streamProcess call at a time takes all the samples queued so far once
// there are at least chunk of them, so a slow pass makes the next call
// longer rather than the queue longer, up to MAX_SAMPLES, which
// streamProcess takes at most; a longer backlog goes in several calls. With
// paced false, every call takes exactly chunk samples (the last one fewer),
// as `whistle stream --chunk`.
// onUpdate gets {committed, pending, language, received, passMs, words,
// last} after every call.
export class StreamSession {
  constructor(
    client,
    { chunk = SAMPLE_RATE, paced = true, onUpdate = () => {} } = {},
  ) {
    this.client = client;
    this.chunk = chunk;
    this.paced = paced;
    this.onUpdate = onUpdate;
    this.queue = [];
    this.queued = 0;
    this.busy = null;
    this.committed = [];
    this.words = [];
    this.calls = [];
    this.error = null;
  }

  start(opts) {
    return this.client.streamStart(opts);
  }

  feed(pcm) {
    this.queue.push(pcm);
    this.queued += pcm.length;
    this.pump();
  }

  // pump starts the next call if there is none running and enough queued;
  // after a failed call it starts none, and stop reports the error.
  pump() {
    if (this.busy || this.error || this.queued < this.chunk) {
      return this.busy;
    }

    const n = this.paced ? Math.min(this.queued, MAX_SAMPLES) : this.chunk;
    this.busy = this.client
      .streamProcess(this.take(n))
      .then((r) => this.update(r))
      .catch((err) => {
        this.error = err;
      })
      .finally(() => {
        this.busy = null;
      })
      .then(() => (this.error ? undefined : this.pump()));
    return this.busy;
  }

  take(n) {
    const out = new Float32Array(n);
    let at = 0;
    while (at < n) {
      const head = this.queue[0];
      const k = Math.min(head.length, n - at);
      out.set(head.subarray(0, k), at);
      at += k;
      if (k === head.length) {
        this.queue.shift();
      } else {
        this.queue[0] = head.subarray(k);
      }
    }
    this.queued -= n;
    return out;
  }

  update(r) {
    this.calls.push(r);
    if (r.text) {
      this.committed.push(r.text);
    }
    this.words.push(...r.words);
    this.onUpdate({
      committed: this.committed.join(" "),
      pending: r.pending,
      language: r.language,
      received: r.received,
      passMs: r.pass_ms,
      words: this.words,
      last: r,
    });
    return r;
  }

  // stop feeds what is left (in chunks, of at most MAX_SAMPLES if paced),
  // waits for the passes and ends the stream with streamStop, which commits
  // the pending words. It resolves to the committed text.
  async stop() {
    for (;;) {
      await this.busy;
      if (this.error) {
        break;
      }
      if (this.queued === 0) {
        break;
      }
      const n = Math.min(this.paced ? MAX_SAMPLES : this.chunk, this.queued);
      await this.client
        .streamProcess(this.take(n))
        .then((r) => this.update(r))
        .catch((err) => {
          this.error = err;
        });
    }

    const r = await this.client.streamStop();
    this.update(r);
    if (this.error) {
      throw this.error;
    }
    return this.committed.join(" ");
  }
}
