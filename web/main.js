// The demo page: model, options, file transcription, streaming of a file
// and of the microphone. The model runs in worker.js; whistle-client.js
// holds everything that is not DOM.
import {
  PINNED_MODEL,
  SAMPLE_RATE,
  StreamSession,
  WhistleClient,
  decodeAudio,
  fetchModel,
  sha256Hex,
} from "./whistle-client.js";

const $ = (id) => document.getElementById(id);

const client = new WhistleClient(new Worker("worker.js"));

let modelReady = false;
let audio = null; // {name, buffer} of the picked file
let busy = false;
let mic = null; // the running microphone stream

$("model-url").value = PINNED_MODEL.url;
$("model-sha").value = PINNED_MODEL.sha256;

client.ready.then(
  (version) => {
    $("runtime").textContent =
      `WebAssembly module ${version} running in a worker.`;
  },
  (err) => {
    $("runtime").textContent = `The WebAssembly module failed: ${err.message}`;
    $("runtime").classList.add("error");
  },
);

function status(id, text, error = false) {
  const el = $(id);
  el.textContent = text;
  el.classList.toggle("error", error);
}

function mib(n) {
  return `${(n / 1048576).toFixed(1)} MiB`;
}

function refresh() {
  $("transcribe").disabled = !modelReady || !audio || busy;
  $("stream-file").disabled = !modelReady || !audio || busy;
  $("mic").disabled = !modelReady || (busy && !mic);
  $("model-fetch").disabled = busy;
  $("model-file").disabled = busy;
}

async function withBusy(fn) {
  busy = true;
  refresh();
  try {
    await fn();
  } finally {
    busy = false;
    refresh();
  }
}

// --- model ---------------------------------------------------------------

async function loadModel(buffer, origin) {
  const size = buffer.byteLength;
  status("model-status", `Loading ${origin} (${mib(size)}) …`);
  const t0 = performance.now();
  const info = await client.load(buffer);
  const ms = performance.now() - t0;

  const lang = $("language");
  lang.length = 1;
  for (const code of info.languages) {
    lang.add(new Option(code, code));
  }

  const depth = $("depth");
  depth.length = 1;
  for (let n = 2; n < 8; n++) {
    depth.add(new Option(`${n} layers`, String(n)));
  }

  modelReady = true;
  status(
    "model-status",
    `Model ready: ${origin}, ${mib(size)}, loaded in ${ms.toFixed(0)} ms.`,
  );
}

$("model-fetch").addEventListener("click", () =>
  withBusy(async () => {
    const url = $("model-url").value.trim();
    const sha256 = $("model-sha").value.trim().toLowerCase();
    const bar = $("model-progress");
    try {
      bar.hidden = false;
      bar.value = 0;
      status("model-status", `Fetching ${url} …`);
      const got = await fetchModel(url, {
        sha256,
        onProgress: (loaded, total) => {
          bar.value = total ? loaded / total : 0;
          status(
            "model-status",
            `Fetching … ${mib(loaded)}${total ? ` of ${mib(total)}` : ""}`,
          );
        },
      });
      bar.value = 1;
      const check = sha256 ? "SHA-256 checked" : `SHA-256 ${got.sha256}`;
      await loadModel(
        got.buffer,
        `${got.cached ? "the cached copy" : "the download"}, ${check}`,
      );
    } catch (err) {
      status("model-status", err.message, true);
    } finally {
      bar.hidden = true;
    }
  }),
);

$("model-file").addEventListener("change", (event) =>
  withBusy(async () => {
    const file = event.target.files[0];
    if (!file) {
      return;
    }
    try {
      const buffer = await file.arrayBuffer();
      const hash = await sha256Hex(buffer);
      const pinned =
        hash === PINNED_MODEL.sha256
          ? "the pinned file"
          : `not the pinned file, SHA-256 ${hash}`;
      await loadModel(buffer, `${file.name} (${pinned})`);
    } catch (err) {
      status("model-status", err.message, true);
    }
  }),
);

// --- options -------------------------------------------------------------

function transcribeOptions() {
  return {
    language: $("language").value,
    keywords: keywords(),
    word_timestamps: $("word-timestamps").checked,
    depth: Number($("depth").value),
  };
}

function streamOptions() {
  return { language: $("language").value, keywords: keywords() };
}

function keywords() {
  return $("keywords")
    .value.split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

// --- result --------------------------------------------------------------

function showResult(
  title,
  { committed, pending = "", meta = "", words = null, json = "" },
) {
  $("result-section").hidden = false;
  $("result-title").textContent = title;
  $("result-text").querySelector(".committed").textContent = committed;
  $("result-text").querySelector(".pending").textContent = pending;
  $("result-meta").textContent = meta;
  $("result-json").textContent = json;

  const table = $("words");
  const body = table.querySelector("tbody");
  body.replaceChildren();
  table.hidden = !words || words.length === 0;
  for (const w of words || []) {
    const tr = document.createElement("tr");
    for (const v of [
      w.word,
      w.start.toFixed(2),
      w.end.toFixed(2),
      w.probability.toFixed(3),
    ]) {
      const td = document.createElement("td");
      td.textContent = v;
      tr.append(td);
    }
    body.append(tr);
  }
}

// --- file ----------------------------------------------------------------

$("audio-file").addEventListener("change", async (event) => {
  const file = event.target.files[0];
  audio = file ? { name: file.name, buffer: await file.arrayBuffer() } : null;
  $("audio-name").textContent = file ? `${file.name}, ${mib(file.size)}` : "";
  status("file-status", "");
  refresh();
});

async function decodePicked() {
  const { pcm, decoder, reason } = await decodeAudio(client, audio.buffer);
  const how =
    decoder === "go"
      ? "read by the CLI's WAV reader"
      : `decoded and resampled by the browser (${reason}), so it may differ slightly from whistle transcribe`;
  return { pcm, how };
}

$("transcribe").addEventListener("click", () =>
  withBusy(async () => {
    try {
      status("file-status", "Decoding …");
      const { pcm, how } = await decodePicked();
      const seconds = pcm.length / SAMPLE_RATE;
      status("file-status", `Transcribing ${seconds.toFixed(2)} s (${how}) …`);
      const t0 = performance.now();
      const r = await client.transcribe(pcm, transcribeOptions());
      const wall = performance.now() - t0;
      showResult(`Transcript of ${audio.name}`, {
        committed: r.text || "(silence)",
        meta:
          `language ${r.language || "–"} · ${seconds.toFixed(2)} s of audio · ttft ${r.ttft_ms.toFixed(1)} ms · ` +
          `${r.decode_tps.toFixed(1)} tok/s · ${(wall / 1000).toFixed(2)} s in all`,
        words: r.words,
        json: r.json,
      });
      status("file-status", `Done; the samples were ${how}.`);
    } catch (err) {
      status("file-status", err.message, true);
    }
  }),
);

function streamView(title) {
  return (u) =>
    showResult(title, {
      committed: u.committed,
      pending: u.pending,
      meta: `language ${u.language || "–"} · ${u.received.toFixed(2)} s received · last pass ${u.passMs.toFixed(0)} ms`,
      words: u.words,
      json: u.last.json,
    });
}

$("stream-file").addEventListener("click", () =>
  withBusy(async () => {
    try {
      status("file-status", "Decoding …");
      const { pcm, how } = await decodePicked();
      status(
        "file-status",
        `Streaming ${(pcm.length / SAMPLE_RATE).toFixed(2)} s in 1 s chunks (${how}) …`,
      );
      const s = new StreamSession(client, {
        paced: false,
        onUpdate: streamView(`Stream of ${audio.name}`),
      });
      await s.start(streamOptions());
      s.feed(pcm);
      await s.stop();
      status(
        "file-status",
        `Streamed in ${s.calls.length - 1} calls and a stop, as whistle stream --chunk 1 does.`,
      );
    } catch (err) {
      status("file-status", err.message, true);
    }
  }),
);

// --- microphone ----------------------------------------------------------

async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    },
  });
  const ctx = new AudioContext();

  try {
    await ctx.audioWorklet.addModule("recorder-worklet.js");
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "whistle-recorder", {
      numberOfOutputs: 1,
    });
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node).connect(mute).connect(ctx.destination);

    const session = new StreamSession(client, {
      onUpdate: streamView("Microphone"),
    });
    await session.start(streamOptions());

    const level = $("mic-level").firstElementChild;
    const flushed = { resolve: () => {} };
    node.port.onmessage = (event) => {
      const block = event.data;
      if (block === "flushed") {
        flushed.resolve();
        return;
      }
      let peak = 0;
      for (const v of block) {
        peak = Math.max(peak, Math.abs(v));
      }
      level.style.width = `${Math.min(100, peak * 200)}%`;
      session.feed(block);
    };

    showResult("Microphone", { committed: "", pending: "listening …" });
    return { stream, ctx, node, session, flushed, rate: ctx.sampleRate };
  } catch (err) {
    stream.getTracks().forEach((t) => t.stop());
    ctx.close().catch(() => {});
    throw err;
  }
}

// stopMic takes the worklet's last partial block, releases the microphone
// and ends the stream, which commits the pending words.
async function stopMic(m) {
  try {
    await new Promise((resolve) => {
      m.flushed.resolve = resolve;
      m.node.port.postMessage("flush");
      setTimeout(resolve, 500); // a closed context never answers
    });
  } finally {
    m.node.port.onmessage = null;
    m.stream.getTracks().forEach((t) => t.stop());
    m.ctx.close().catch(() => {});
    $("mic-level").firstElementChild.style.width = "0";
  }
  return m.session.stop();
}

$("mic").addEventListener("click", async () => {
  const button = $("mic");
  if (!mic) {
    button.disabled = true;
    try {
      busy = true;
      mic = await startMic();
      button.textContent = "Stop microphone";
      status(
        "mic-status",
        `Listening at ${mic.rate} Hz, resampled to 16 kHz; a pass runs on every second of audio.`,
      );
    } catch (err) {
      busy = false;
      mic = null;
      status("mic-status", err.message, true);
    }
    refresh();
    return;
  }

  const m = mic;
  button.disabled = true;
  status("mic-status", "Finishing the last pass …");
  try {
    await stopMic(m);
    status("mic-status", "Stopped; the pending words are committed.");
  } catch (err) {
    status("mic-status", err.message, true);
  } finally {
    mic = null;
    busy = false;
    button.textContent = "Start microphone";
    refresh();
  }
});

refresh();
