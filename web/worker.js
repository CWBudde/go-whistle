// The Web Worker that runs whistle.wasm, so that a transcription does not
// block the page. It takes {id, op, args} messages, runs them one at a time
// on globalThis.whistle (cmd/whistle-wasm) and answers {id, result} or
// {id, error}. sha256 gives its buffer back along with the digest, as
// {id, result: {sha256, buffer}}, so that the page need not copy the bytes
// it hashes. Once the module runs it posts {type: "ready", version}, or
// {type: "fatal", error} when it cannot start.
/* global Go */
"use strict";

importScripts("wasm_exec.js");

const ops = new Set([
  "load",
  "sha256",
  "decodeWAV",
  "transcribe",
  "streamStart",
  "streamProcess",
  "streamStop",
]);

async function boot() {
  const go = new Go();
  const res = await fetch(new URL("whistle.wasm", self.location.href));
  if (!res.ok) {
    throw new Error(
      `whistle.wasm: HTTP ${res.status} (build it with: just wasm)`,
    );
  }

  let instance;
  if (res.headers.get("content-type") === "application/wasm") {
    ({ instance } = await WebAssembly.instantiateStreaming(
      res,
      go.importObject,
    ));
  } else {
    ({ instance } = await WebAssembly.instantiate(
      await res.arrayBuffer(),
      go.importObject,
    ));
  }

  // main installs globalThis.whistle before it blocks, so it is there once
  // run returns control; the promise run returns settles only on exit.
  go.run(instance).then(() => {
    self.postMessage({ type: "fatal", error: "whistle.wasm exited" });
  });

  if (!self.whistle) {
    throw new Error("whistle.wasm did not install globalThis.whistle");
  }

  return self.whistle;
}

const booted = boot();

booted.then(
  (api) => self.postMessage({ type: "ready", version: api.version }),
  (err) =>
    self.postMessage({
      type: "fatal",
      error: String((err && err.message) || err),
    }),
);

// The calls run one after the other, in the order they came.
let queue = Promise.resolve();

self.onmessage = (event) => {
  const { id, op, args = [] } = event.data;

  queue = queue.then(async () => {
    try {
      if (!ops.has(op)) {
        throw new Error(`unknown call ${op}`);
      }

      const api = await booted;
      if (op === "sha256") {
        const [buffer] = args;
        const sha256 = await api.sha256(buffer);
        self.postMessage({ id, result: { sha256, buffer } }, [buffer]);
        return;
      }

      const result = await api[op](...args);
      const transfer = result instanceof Float32Array ? [result.buffer] : [];
      self.postMessage({ id, result }, transfer);
    } catch (err) {
      self.postMessage({ id, error: String((err && err.message) || err) });
    }
  });
};
