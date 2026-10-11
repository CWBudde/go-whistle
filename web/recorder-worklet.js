// The microphone's AudioWorklet: it mixes the input to mono, resamples it
// to 16 kHz and posts Float32Array blocks of 100 ms to the page; on the
// message "flush" it posts the partial block and then "flushed".
/* global AudioWorkletProcessor, registerProcessor, sampleRate */
"use strict";

const TARGET_RATE = 16000;
const BLOCK = TARGET_RATE / 10;

// Resampler converts a stream from rate to TARGET_RATE: a Blackman-windowed
// sinc low-pass with its cutoff at 0.45 · TARGET_RATE and 16 zero crossings
// on each side, evaluated at every output instant. Output n sits at input
// position n · rate / TARGET_RATE, whose fraction takes only TARGET_RATE /
// gcd(rate, TARGET_RATE) values (1 at 48 kHz, 160 at 44.1 kHz), so the
// weights of each come from a table. At TARGET_RATE it passes the samples
// through.
class Resampler {
  constructor(rate) {
    this.rate = rate;
    this.fc = (0.45 * Math.min(rate, TARGET_RATE)) / rate; // cutoff in cycles per input sample
    this.half = Math.ceil(16 / (2 * this.fc)); // taps on each side
    this.taps = 2 * this.half;
    this.unit = gcd(rate, TARGET_RATE);

    const phases = TARGET_RATE / this.unit;
    this.table = null;
    if (phases * this.taps <= 1 << 20) {
      this.table = new Float32Array(phases * this.taps);
      for (let p = 0; p < phases; p++) {
        const f = (p * this.unit) / TARGET_RATE;
        for (let k = 1 - this.half; k <= this.half; k++) {
          this.table[p * this.taps + k + this.half - 1] = this.kernel(k - f);
        }
      }
    }

    this.buf = new Float32Array(0);
    this.base = 0; // the input index of buf[0]
    this.n = 0; // the next output sample
  }

  // kernel is the filter's weight of an input sample d samples from the
  // output instant.
  kernel(d) {
    const u = d / this.half;
    if (Math.abs(u) >= 1) {
      return 0;
    }
    const w =
      0.42 + 0.5 * Math.cos(Math.PI * u) + 0.08 * Math.cos(2 * Math.PI * u);
    const arg = 2 * this.fc * d;
    const sinc = arg === 0 ? 1 : Math.sin(Math.PI * arg) / (Math.PI * arg);
    return 2 * this.fc * sinc * w;
  }

  // push takes input samples and returns the output samples they complete.
  push(x) {
    if (this.rate === TARGET_RATE) {
      return x.slice();
    }

    const joined = new Float32Array(this.buf.length + x.length);
    joined.set(this.buf);
    joined.set(x, this.buf.length);
    this.buf = joined;

    const end = this.base + this.buf.length;
    const out = [];
    for (;;) {
      const pos = this.n * this.rate; // the input position times TARGET_RATE
      const i0 = Math.floor(pos / TARGET_RATE);
      if (i0 + this.half >= end) {
        const keep = Math.max(0, i0 - this.half + 1 - this.base);
        this.buf = this.buf.slice(keep);
        this.base += keep;
        return Float32Array.from(out);
      }
      out.push(this.sample(i0, pos - i0 * TARGET_RATE));
      this.n++;
    }
  }

  // sample is the output at input position i0 + phase / TARGET_RATE.
  sample(i0, phase) {
    const row = (phase / this.unit) * this.taps + this.half - 1;
    const f = phase / TARGET_RATE;
    let sum = 0;
    for (let k = 1 - this.half; k <= this.half; k++) {
      const j = i0 + k - this.base;
      if (j < 0) {
        continue; // before the first sample: silence
      }
      sum +=
        this.buf[j] * (this.table ? this.table[row + k] : this.kernel(k - f));
    }
    return sum;
  }
}

function gcd(a, b) {
  while (b) {
    [a, b] = [b, a % b];
  }
  return a;
}

if (typeof AudioWorkletProcessor !== "undefined") {
  class WhistleRecorder extends AudioWorkletProcessor {
    constructor() {
      super();
      this.resampler = new Resampler(sampleRate);
      this.block = new Float32Array(BLOCK);
      this.filled = 0;

      // "flush" posts the partial block, then "flushed": the page stops
      // the microphone without losing its last 100 ms.
      this.port.onmessage = (event) => {
        if (event.data !== "flush") {
          return;
        }
        if (this.filled > 0) {
          const rest = this.block.slice(0, this.filled);
          this.port.postMessage(rest, [rest.buffer]);
          this.filled = 0;
        }
        this.port.postMessage("flushed");
      };
    }

    process(inputs) {
      const channels = inputs[0];
      if (!channels || channels.length === 0) {
        return true;
      }

      const n = channels[0].length;
      const mono = new Float32Array(n);
      for (const ch of channels) {
        for (let i = 0; i < n; i++) {
          mono[i] += ch[i] / channels.length;
        }
      }

      for (const v of this.resampler.push(mono)) {
        this.block[this.filled++] = v;
        if (this.filled === BLOCK) {
          this.port.postMessage(this.block, [this.block.buffer]);
          this.block = new Float32Array(BLOCK);
          this.filled = 0;
        }
      }
      return true;
    }
  }

  registerProcessor("whistle-recorder", WhistleRecorder);
}
