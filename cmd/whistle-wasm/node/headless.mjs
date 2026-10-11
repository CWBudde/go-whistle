// Headless test of the web demo under Node: it runs web/worker.js (and so
// web/whistle.wasm) in this process as the page's worker, drives it through
// web/whistle-client.js as web/main.js does, and compares the results with
// the native CLI's. It also checks the microphone resampler of
// web/recorder-worklet.js. Run by `just wasm-test`:
//
//   node headless.mjs --web web [--model models/whistle/whistle.cact --cli bin/whistle --audio testdata/audio CLIP...]
//
// Without --model, or when the model file is missing, only the resampler and
// the client's model-free parts are checked.
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
// goes in calls of at most MAX_SAMPLES, and a cache write that fails does
// not fail fetchModel.
async function checkClient() {
  const { StreamSession, MAX_SAMPLES, fetchModel, sha256Hex } = await import(
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
      return { text: "", words: [], pending: "" };
    },
    streamStop: async () => ({ text: "", words: [], pending: "" }),
  };
  const s = new StreamSession(fake);
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

  // A cache whose writes fail, as over the origin's quota.
  const bytes = new Uint8Array([1, 2, 3, 4]);
  const sha256 = await sha256Hex(bytes.slice().buffer);
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
  } finally {
    globalThis.caches = caches;
  }
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

async function checkModel() {
  const {
    WhistleClient,
    StreamSession,
    PINNED_MODEL,
    decodeAudio,
    fetchModel,
  } = await import(pathToFileURL(resolve(web, "whistle-client.js")).href);

  const client = new WhistleClient(startWorker());
  const version = await client.ready;

  // The page's fetch path, on the local file: the stream read and the
  // SHA-256 check against the pinned hash.
  const model = resolve(opts.model);
  const t0 = performance.now();
  const got = await fetchModel(PINNED_MODEL.url, {
    sha256: PINNED_MODEL.sha256,
    fetchFn: async () => new Response(readFileSync(model)),
  });
  const info = await client.load(got.buffer);
  console.log(
    `whistle.wasm ${version}: model ${info.bytes} bytes, SHA-256 checked, loaded in ` +
      `${(performance.now() - t0).toFixed(0)} ms; languages ${info.languages.join(" ")}`,
  );

  const cli = (...args) =>
    execFileSync(resolve(opts.cli), [...args, "--model", model], {
      encoding: "utf8",
    });
  const wav = (id) => readFileSync(resolve(opts.audio, `${id}.wav`));
  const buffer = (b) =>
    b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);

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
}

checkResampler();
await checkClient();

if (opts.model && existsSync(opts.model)) {
  await checkModel();
} else {
  console.log(
    opts.model
      ? `SKIP the model checks: ${opts.model} missing; run \`whistle download\``
      : "SKIP the model checks: no --model",
  );
}

console.log(failures === 0 ? "PASS" : `FAIL: ${failures} check(s)`);
process.exit(failures === 0 ? 0 : 1);
