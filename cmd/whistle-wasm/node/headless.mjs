// Headless test of the web demo under Node: it runs web/worker.js (and so
// web/whistle.wasm) in this process as the page's worker, drives it through
// web/whistle-client.js as web/main.js does, and compares the results with
// the native CLI's. It also checks the microphone resampler of
// web/recorder-worklet.js, and runs the page's model path once more as in an
// insecure context (a page over plain http), without Web Crypto and Cache
// Storage, where Go hashes the model. Run by `just wasm-test`:
//
//   node headless.mjs --web web [--model models/whistle/whistle.cact --cli bin/whistle --audio testdata/audio CLIP...]
//
// Without --model, or when the model file is missing, only the resampler and
// the client's and the worker's model-free parts are checked.
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";
import vm from "node:vm";

const { values: opts, positionals: clips } = parseArgs({
  options: {
    web: { type: "string", default: "web" },
    model: { type: "string" },
    cli: { type: "string" },
    audio: { type: "string", default: "testdata/audio" },
    stream: { type: "string", default: "ls-clean-6930-75918-0000" },
  },
  allowPositionals: true,
});

const web = resolve(opts.web);
let failures = 0;

function fail(msg) {
  failures++;
  console.log(`FAIL ${msg}`);
}

// --- the resampler -------------------------------------------------------

function checkResampler() {
  const src = readFileSync(resolve(web, "recorder-worklet.js"), "utf8");
  const Resampler = vm.runInNewContext(`${src}\n;Resampler`, {});

  for (const rate of [16000, 22050, 44100, 48000, 96000]) {
    const r = new Resampler(rate);
    const t0 = performance.now();
    const tone = (f, n0, n) =>
      Float32Array.from(
        { length: n },
        (_, i) => 0.5 * Math.sin((2 * Math.PI * f * (n0 + i)) / rate),
      );

    // 2 s of a 440 Hz tone in 128-sample render quanta, then 2 s of 12 kHz,
    // which has no place below 8 kHz and must vanish rather than alias
    // wherever the input rate can hold it.
    const out = [];
    for (let n = 0; n < 4 * rate; n += 128) {
      const f = n < 2 * rate ? 440 : 12000;
      out.push(...r.push(tone(f, n, Math.min(128, 4 * rate - n))));
    }

    const ms = performance.now() - t0;

    let passErr = 0;
    for (let i = 1600; i < 30400; i++) {
      passErr = Math.max(
        passErr,
        Math.abs(out[i] - 0.5 * Math.sin((2 * Math.PI * 440 * i) / 16000)),
      );
    }

    let stopPeak = 0;
    for (let i = 33600; i < out.length; i++) {
      stopPeak = Math.max(stopPeak, Math.abs(out[i]));
    }

    const want = 4 * 16000;
    const lag = want - out.length;
    console.log(
      `resampler ${rate} Hz: ${out.length} samples (${lag} held back), 440 Hz error ${passErr.toExponential(1)}, ` +
        `12 kHz peak ${stopPeak.toExponential(1)}, ${(ms / 4).toFixed(1)} ms per second of audio`,
    );
    if (
      lag < 0 ||
      lag > 64 ||
      passErr > 2e-3 ||
      (rate > 2 * 12000 && stopPeak > 1e-3)
    ) {
      fail(`resampler ${rate} Hz`);
    }
  }
}

// --- the client without a model ----------------------------------------

// checkClient runs whistle-client.js against fakes: a paced stream's backlog
// goes in calls of at most MAX_SAMPLES, a failed call stops the pumping,
// and a cache that fails to open, read or write does not fail fetchModel.
async function checkClient() {
  const { StreamSession, MAX_SAMPLES, fetchModel, sha256Of } = await import(
    pathToFileURL(resolve(web, "whistle-client.js")).href
  );

  // A slow first pass lets 70 s pile up behind it.
  const sizes = [];
  let release;
  const first = new Promise((r) => (release = r));
  const fake = {
    streamStart: async () => undefined,
    async streamProcess(pcm) {
      sizes.push(pcm.length);
      if (pcm.length > MAX_SAMPLES) {
        throw new Error(`${pcm.length} samples, over ${MAX_SAMPLES}`);
      }
      if (sizes.length === 1) {
        await first;
      }
      return { text: "", words: [], pending: "", pass_ms: sizes.length };
    },
    streamStop: async () => ({ text: "", words: [], pending: "", pass_ms: 0 }),
  };
  let last = null;
  const s = new StreamSession(fake, { onUpdate: (u) => (last = u) });
  await s.start({});
  s.feed(new Float32Array(16000));
  for (let k = 0; k < 70; k++) {
    s.feed(new Float32Array(16000));
  }
  release();
  try {
    await s.stop();
  } catch (err) {
    fail(`client: paced stream: ${err.message}`);
  }
  const total = sizes.reduce((a, b) => a + b, 0);
  console.log(
    `client: paced backlog of 70 s in calls of ${sizes.join(", ")} samples`,
  );
  if (total !== 71 * 16000 || sizes.some((n) => n > MAX_SAMPLES)) {
    fail(`client: paced calls ${sizes.join(", ")}`);
  }

  // The stop runs no pass: after it, the update still has the last pass's
  // time, not the stop's pass_ms of 0.
  const n = sizes.length;
  console.log(
    `client: after the stop: ${last.passes} passes, the last ${last.passMs} ms, ${last.totalMs} ms in all`,
  );
  if (
    last.passes !== n ||
    last.passMs !== n ||
    last.totalMs !== (n * (n + 1)) / 2
  ) {
    fail(
      `client: after the stop: ${last.passes} passes, last ${last.passMs} ms, ${last.totalMs} ms; want ${n}, ${n}, ${(n * (n + 1)) / 2}`,
    );
  }

  // After a failed call, more audio starts no further calls, and stop
  // rejects with the error.
  let calls = 0;
  const failing = {
    ...fake,
    async streamProcess() {
      calls++;
      throw new Error("pass failed");
    },
  };
  const f = new StreamSession(failing);
  await f.start({});
  f.feed(new Float32Array(16000));
  await f.busy;
  for (let k = 0; k < 5; k++) {
    f.feed(new Float32Array(16000));
  }
  const stopped = await f.stop().then(
    () => "resolved",
    (err) => err.message,
  );
  console.log(
    `client: a failed call, 5 s more audio: ${calls} call(s), stop ${stopped}`,
  );
  if (calls !== 1 || stopped !== "pass failed") {
    fail(`client: after a failed call: ${calls} calls, stop ${stopped}`);
  }

  // A cache whose writes fail, as over the origin's quota.
  const bytes = new Uint8Array([1, 2, 3, 4]);
  const { sha256 } = await sha256Of(bytes.slice().buffer);
  const caches = globalThis.caches;
  globalThis.caches = {
    open: async () => ({
      match: async () => undefined,
      delete: async () => false,
      put: async () => {
        throw new DOMException("quota", "QuotaExceededError");
      },
    }),
  };
  try {
    const got = await fetchModel("https://example.invalid/whistle.cact", {
      sha256,
      fetchFn: async () => new Response(bytes),
    });
    console.log(
      `client: fetchModel with a failing cache write: ${got.buffer.byteLength} bytes`,
    );
    if (got.buffer.byteLength !== bytes.length || got.cached) {
      fail("client: fetchModel with a failing cache write");
    }
  } catch (err) {
    fail(`client: fetchModel with a failing cache write: ${err.message}`);
  }

  // Cache Storage that cannot be opened (storage disabled or denied), and
  // one that cannot be read.
  const broken = {
    "cannot be opened": {
      open: async () => {
        throw new DOMException("denied", "SecurityError");
      },
    },
    "cannot be read": {
      open: async () => ({
        match: async () => {
          throw new DOMException("denied", "SecurityError");
        },
        put: async () => undefined,
      }),
    },
  };
  for (const [what, stub] of Object.entries(broken)) {
    globalThis.caches = stub;
    try {
      const got = await fetchModel("https://example.invalid/whistle.cact", {
        sha256,
        fetchFn: async () => new Response(bytes),
      });
      console.log(
        `client: fetchModel with a cache that ${what}: ${got.buffer.byteLength} bytes`,
      );
      if (got.buffer.byteLength !== bytes.length || got.cached) {
        fail(`client: fetchModel with a cache that ${what}`);
      }
    } catch (err) {
      fail(`client: fetchModel with a cache that ${what}: ${err.message}`);
    }
  }
  globalThis.caches = caches;
}

// --- the worker in this process ------------------------------------------

// startWorker runs worker.js in this realm, as a dedicated worker would,
// and returns the object WhistleClient talks to. Messages are structured
// clones with their buffers transferred, delivered asynchronously.
function startWorker() {
  const page = {
    onmessage: null,
    onerror: null,
    postMessage(msg, transfer = []) {
      const data = structuredClone(msg, { transfer });
      setImmediate(() => globalThis.onmessage({ data }));
    },
  };

  globalThis.self = globalThis;
  globalThis.location = { href: pathToFileURL(resolve(web, "worker.js")).href };
  globalThis.postMessage = (msg, transfer = []) => {
    const data = structuredClone(msg, { transfer });
    setImmediate(() => page.onmessage({ data }));
  };
  globalThis.importScripts = (...names) => {
    for (const name of names) {
      const file = resolve(web, name);
      vm.runInThisContext(readFileSync(file, "utf8"), { filename: file });
    }
  };

  const netFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const u = new URL(url);
    if (u.protocol !== "file:") {
      return netFetch(url, init);
    }
    return new Response(readFileSync(u), {
      headers: { "content-type": "application/wasm" },
    });
  };

  const file = resolve(web, "worker.js");
  vm.runInThisContext(readFileSync(file, "utf8"), { filename: file });

  return page;
}

// --- an insecure context ------------------------------------------------

// insecure runs fn as on a page served over plain http, other than from
// localhost, which is no secure context: crypto has getRandomValues (Go's
// runtime needs it) but no subtle, and there is no Cache Storage.
async function insecure(fn) {
  const saved = ["crypto", "caches"].map((name) => [
    name,
    Object.getOwnPropertyDescriptor(globalThis, name),
  ]);
  const webCrypto = globalThis.crypto;
  const define = (name, value) =>
    Object.defineProperty(globalThis, name, {
      value,
      configurable: true,
      enumerable: true,
      writable: true,
    });
  define("crypto", { getRandomValues: (a) => webCrypto.getRandomValues(a) });
  define("caches", undefined);
  try {
    return await fn();
  } finally {
    for (const [name, desc] of saved) {
      if (desc) {
        Object.defineProperty(globalThis, name, desc);
      } else {
        delete globalThis[name];
      }
    }
  }
}

// checkInsecure runs fetchModel without Web Crypto and Cache Storage on 3
// MiB of bytes: the worker hashes them in Go, in several blocks, gives them
// back unchanged, and a wrong hash is rejected; without a worker, the
// SHA-256 fails with a message rather than a TypeError.
async function checkInsecure(client) {
  const { fetchModel, sha256Of } = await import(
    pathToFileURL(resolve(web, "whistle-client.js")).href
  );

  const bytes = new Uint8Array((3 << 20) + 5);
  for (let i = 0; i < bytes.length; i++) {
    bytes[i] = (i * 7) & 255;
  }
  const want = await sha256Of(bytes.slice().buffer);
  const url = "http://192.168.0.2:8080/whistle.cact";
  const fetchFn = async () => new Response(bytes);

  await insecure(async () => {
    const got = await fetchModel(url, { sha256: want.sha256, client, fetchFn });
    const same =
      got.buffer.byteLength === bytes.length &&
      Buffer.compare(new Uint8Array(got.buffer), bytes) === 0;
    const wrong = await fetchModel(url, {
      sha256: "0".repeat(64),
      client,
      fetchFn,
    }).then(
      () => "resolved",
      (err) => err.message,
    );
    const alone = await sha256Of(new ArrayBuffer(1)).then(
      () => "resolved",
      (err) => err.message,
    );
    console.log(
      `insecure context: fetchModel of ${bytes.length} bytes, SHA-256 checked in ${got.hashedIn}, ` +
        `bytes back unchanged ${same}, cached ${got.cached}; a wrong hash: ${wrong}; no worker: ${alone}`,
    );
    if (
      got.hashedIn !== "Go" ||
      got.sha256 !== want.sha256 ||
      want.hashedIn !== "Web Crypto" ||
      !same ||
      got.cached
    ) {
      fail("insecure context: fetchModel");
    }
    if (!wrong.includes(`SHA-256 ${want.sha256}, want 000`)) {
      fail(`insecure context: a wrong hash: ${wrong}`);
    }
    if (!alone.includes("secure context")) {
      fail(`insecure context: no worker: ${alone}`);
    }
  });
}

// timings drops the fields that measure time, which no two runs share.
function timings(json) {
  return json.replace(/,"(ttft_ms|decode_tps|pass_ms)":[0-9.]+/g, "");
}

function sameWords(got, want) {
  if (got.length !== want.length) {
    return false;
  }
  return got.every(
    (w, i) =>
      w.word === want[i].word &&
      w.start === want[i].start &&
      w.end === want[i].end &&
      Math.abs(w.probability - want[i].probability) <= 0.0011,
  );
}

async function checkModel(client, version) {
  const { StreamSession, PINNED_MODEL, decodeAudio, fetchModel } = await import(
    pathToFileURL(resolve(web, "whistle-client.js")).href
  );

  // The page's fetch path, on the local file: the stream read and the
  // SHA-256 check against the pinned hash.
  const model = resolve(opts.model);
  const fetchFn = async () => new Response(readFileSync(model));
  const t0 = performance.now();
  const got = await fetchModel(PINNED_MODEL.url, {
    sha256: PINNED_MODEL.sha256,
    client,
    fetchFn,
  });
  const info = await client.load(got.buffer);
  console.log(
    `whistle.wasm ${version}: model ${info.bytes} bytes, SHA-256 checked in ${got.hashedIn}, loaded in ` +
      `${(performance.now() - t0).toFixed(0)} ms; languages ${info.languages.join(" ")}`,
  );
  if (got.hashedIn !== "Web Crypto") {
    fail(`secure context: SHA-256 checked in ${got.hashedIn}`);
  }

  const cli = (...args) =>
    execFileSync(resolve(opts.cli), [...args, "--model", model], {
      encoding: "utf8",
    });
  const wav = (id) => readFileSync(resolve(opts.audio, `${id}.wav`));
  const buffer = (b) =>
    b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

  const secureJSON = new Map();
  for (const id of clips) {
    const path = resolve(opts.audio, `${id}.wav`);
    const native = JSON.parse(
      cli("transcribe", "--json", "--word-timestamps", path),
    );
    const nativeJSON = cli(
      "transcribe",
      "--json",
      "--word-timestamps",
      path,
    ).trim();

    const { pcm, decoder } = await decodeAudio(client, buffer(wav(id)));
    const t = performance.now();
    const r = await client.transcribe(pcm, { word_timestamps: true });
    const wall = performance.now() - t;

    secureJSON.set(id, timings(r.json));
    const exact = timings(r.json) === timings(nativeJSON);
    const same =
      r.text === native.text &&
      r.language === native.language &&
      sameWords(r.words, native.words);
    console.log(
      `${id}: [${r.language}] ${(pcm.length / 16000).toFixed(2)} s, ${decoder} decoder, wasm ttft ${r.ttft_ms} ms, ` +
        `${r.decode_tps} tok/s, ${wall.toFixed(0)} ms in all; native ttft ${native.ttft_ms} ms, ` +
        `${native.decode_tps} tok/s; equal to whistle transcribe: ${same}, JSON without timings ${exact}`,
    );
    if (decoder !== "go" || !same) {
      fail(`${id}: wasm ${r.json}\n  native ${nativeJSON}`);
    }
  }

  // The file stream: 1 s calls and the stop, as `whistle stream --json`.
  const path = resolve(opts.audio, `${opts.stream}.wav`);
  const native = cli("stream", "--json", path).trim().split("\n");
  const { pcm } = await decodeAudio(client, buffer(wav(opts.stream)));
  const s = new StreamSession(client, { paced: false });
  await s.start({});
  s.feed(pcm);
  const text = await s.stop();
  const calls = s.calls.map((c) => timings(c.json));
  const equal =
    calls.length === native.length &&
    calls.every((c, i) => c === timings(native[i]));
  console.log(
    `${opts.stream} stream: ${calls.length} calls with the stop, "${text}"; equal to whistle stream --json: ${equal}`,
  );
  if (!equal) {
    fail(
      `stream: wasm\n  ${calls.join("\n  ")}\nnative\n  ${native.map(timings).join("\n  ")}`,
    );
  }

  // The stop runs no pass, so the session keeps the last pass's time.
  const lastPass = s.calls.at(-2).pass_ms;
  const total = s.calls.slice(0, -1).reduce((a, c) => a + c.pass_ms, 0);
  console.log(
    `${opts.stream} stream: ${s.passes} passes, the last ${s.passMs} ms, ${s.totalMs.toFixed(1)} ms in all; the stop's pass_ms ${s.calls.at(-1).pass_ms}`,
  );
  if (
    s.passes !== s.calls.length - 1 ||
    s.passMs !== lastPass ||
    !(s.passMs > 0) ||
    Math.abs(s.totalMs - total) > 1e-9
  ) {
    fail(
      `stream: ${s.passes} passes, the last ${s.passMs} ms, ${s.totalMs} ms in all`,
    );
  }

  // The same page path as in an insecure context: Go hashes the model, a
  // wrong hash is rejected, and the first clip transcribes as above.
  if (clips.length === 0) {
    return;
  }
  const id = clips[0];
  await insecure(async () => {
    const t = performance.now();
    const got = await fetchModel(PINNED_MODEL.url, {
      sha256: PINNED_MODEL.sha256,
      client,
      fetchFn,
    });
    const info = await client.load(got.buffer);
    const ms = performance.now() - t;

    const wrongHash = PINNED_MODEL.sha256.replace(/.$/, (c) =>
      c === "0" ? "1" : "0",
    );
    const wrong = await fetchModel(PINNED_MODEL.url, {
      sha256: wrongHash,
      client,
      fetchFn,
    }).then(
      () => "resolved",
      (err) => err.message,
    );

    const { pcm } = await decodeAudio(client, buffer(wav(id)));
    const r = await client.transcribe(pcm, { word_timestamps: true });
    const same = timings(r.json) === secureJSON.get(id);
    console.log(
      `insecure context: model ${info.bytes} bytes, SHA-256 checked in ${got.hashedIn}, cached ${got.cached}, ` +
        `loaded in ${ms.toFixed(0)} ms; a wrong hash: ${wrong.includes("SHA-256") ? "rejected" : wrong}; ` +
        `${id}: "${r.text}", JSON without timings equal to the secure run: ${same}`,
    );
    if (
      got.hashedIn !== "Go" ||
      got.sha256 !== PINNED_MODEL.sha256 ||
      got.cached ||
      !wrong.includes(`want ${wrongHash}`) ||
      !same
    ) {
      fail(
        `insecure context: ${id}: ${r.json}\n  secure ${secureJSON.get(id)}`,
      );
    }
  });
}

checkResampler();
await checkClient();

const { WhistleClient } = await import(
  pathToFileURL(resolve(web, "whistle-client.js")).href
);
const client = new WhistleClient(startWorker());
const version = await client.ready;
await checkInsecure(client);

if (opts.model && existsSync(opts.model)) {
  await checkModel(client, version);
} else {
  console.log(
    opts.model
      ? `SKIP the model checks: ${opts.model} missing; run \`whistle download\``
      : "SKIP the model checks: no --model",
  );
}

console.log(failures === 0 ? "PASS" : `FAIL: ${failures} check(s)`);
process.exit(failures === 0 ? 0 : 1);
