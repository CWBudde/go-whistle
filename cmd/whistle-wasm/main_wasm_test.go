//go:build js && wasm

package main

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall/js"
	"testing"

	"github.com/cwbudde/go-whistle/whistle"
)

// api registers a fresh JS API on its own object, as main does on
// globalThis, and returns it.
func api(t *testing.T) js.Value {
	t.Helper()

	target := js.Global().Get("Object").New()
	register(target, &engine{})

	return target.Get("whistle")
}

// await blocks until promise p settles; the test goroutine waits, so the
// event loop runs the promise's work.
func await(p js.Value) (js.Value, error) {
	type outcome struct {
		v   js.Value
		err error
	}

	ch := make(chan outcome, 1)

	resolved := js.FuncOf(func(_ js.Value, a []js.Value) any {
		ch <- outcome{v: a[0]}

		return nil
	})
	defer resolved.Release()

	rejected := js.FuncOf(func(_ js.Value, a []js.Value) any {
		if !a[0].InstanceOf(js.Global().Get("Error")) {
			ch <- outcome{err: errors.New("rejected with a non-Error: " + a[0].String())}

			return nil
		}

		ch <- outcome{err: errors.New(a[0].Get("message").String())}

		return nil
	})
	defer rejected.Release()

	p.Call("then", resolved, rejected)

	o := <-ch

	return o.v, o.err
}

// fileBytes reads path into a new Uint8Array.
func fileBytes(t *testing.T, path string) js.Value {
	t.Helper()

	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		t.Skipf("%s missing", path)
	}

	if err != nil {
		t.Fatal(err)
	}

	u8 := js.Global().Get("Uint8Array").New(len(data))
	js.CopyBytesToJS(u8, data)

	return u8
}

// The API rejects with an Error naming the problem: a call out of order or
// a wrong argument.
func TestAPIRejects(t *testing.T) {
	w := api(t)
	pcm := js.Global().Get("Float32Array").New(1600)

	for _, c := range []struct {
		name string
		p    js.Value
		want string
	}{
		{"transcribe before load", w.Call("transcribe", pcm), "call load first"},
		{"transcribe of an Array", w.Call("transcribe", js.ValueOf([]any{0.1})), "Float32Array"},
		{"transcribe without samples", w.Call("transcribe"), "Float32Array"},
		{"load of a string", w.Call("load", "whistle.cact"), "Uint8Array"},
		{"load of garbage", w.Call("load", js.Global().Get("Uint8Array").New(16)), "whistle: "},
		{"decodeWAV of garbage", w.Call("decodeWAV", js.Global().Get("ArrayBuffer").New(16)), "whistle: "},
		{"streamStart before load", w.Call("streamStart"), "call load first"},
		{"streamProcess before streamStart", w.Call("streamProcess", pcm), "call streamStart first"},
		{"streamStop before streamStart", w.Call("streamStop"), "call streamStart first"},
	} {
		_, err := await(c.p)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: %v, want a rejection mentioning %q", c.name, err, c.want)
		}
	}

	if w.Get("sampleRate").Int() != whistle.SampleRate || w.Get("maxSamples").Int() != whistle.MaxSamples {
		t.Errorf("sampleRate %v, maxSamples %v", w.Get("sampleRate"), w.Get("maxSamples"))
	}
}

// decodeWAV hands back a Float32Array of the samples the CLI reads.
func TestAPIDecodeWAV(t *testing.T) {
	path := filepath.Join(clipDir, equalClips[0]+".wav")

	v, err := await(api(t).Call("decodeWAV", fileBytes(t, path)))
	if err != nil {
		t.Fatal(err)
	}

	got, err := float32sArg([]js.Value{v}, 0)
	if err != nil {
		t.Fatal(err)
	}

	data, _ := os.ReadFile(path)

	want, err := decodeWAV(data)
	if err != nil {
		t.Fatal(err)
	}

	if len(got) != len(want) {
		t.Fatalf("%d samples, want %d", len(got), len(want))
	}

	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("sample %d: %v, want %v", i, got[i], want[i])
		}
	}
}

// The worker's path through the JS API: load the model bytes, decode each
// WAV, transcribe it with word timestamps, and stream a clip in 1 s chunks;
// the transcripts are the engine's.
func TestAPITranscribeAndStream(t *testing.T) {
	model := fileBytes(t, cactPath)
	w := api(t)

	info, err := await(w.Call("load", model))
	if err != nil {
		t.Fatal(err)
	}

	if n := info.Get("languages").Length(); n != 7 || info.Get("bytes").Int() != model.Length() {
		t.Errorf("load: %d languages, %v bytes", n, info.Get("bytes"))
	}

	opts := js.ValueOf(map[string]any{"word_timestamps": true})

	for _, id := range equalClips {
		pcm, err := await(w.Call("decodeWAV", fileBytes(t, filepath.Join(clipDir, id+".wav"))))
		if err != nil {
			t.Fatal(err)
		}

		out, err := await(w.Call("transcribe", pcm, opts))
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}

		if out.Type() != js.TypeString {
			t.Fatalf("%s: transcribe resolved to a %v, want the JSON string", id, out.Type())
		}

		res := checkResult(t, id, []byte(out.String()))
		t.Logf("%s: [%s] ttft %.1f ms, %.1f tok/s", id, res.Language, res.TTFT, res.DecodeTPS)
	}

	_, err = await(w.Call("transcribe", js.Global().Get("Float32Array").New(1600), map[string]any{"word_timestamp": true}))
	if err == nil || !strings.Contains(err.Error(), "word_timestamp") {
		t.Errorf("a misspelt option: %v, want a rejection naming it", err)
	}

	pcm, err := await(w.Call("decodeWAV", fileBytes(t, filepath.Join(clipDir, streamClip+".wav"))))
	if err != nil {
		t.Fatal(err)
	}

	_, err = await(w.Call("streamStart", map[string]any{"keywords": []any{}}))
	if err != nil {
		t.Fatal(err)
	}

	var text []string

	commit := func(out js.Value) {
		r := js.Global().Get("JSON").Call("parse", out)
		if s := r.Get("text").String(); s != "" {
			text = append(text, s)
		}
	}

	for i := 0; i < pcm.Length(); i += whistle.SampleRate {
		chunk := pcm.Call("subarray", i, min(i+whistle.SampleRate, pcm.Length()))

		out, err := await(w.Call("streamProcess", chunk))
		if err != nil {
			t.Fatal(err)
		}

		commit(out)
	}

	out, err := await(w.Call("streamStop"))
	if err != nil {
		t.Fatal(err)
	}

	commit(out)

	if got := strings.Join(text, " "); got != streamText {
		t.Errorf("stream text %q, the engine's stream %q", got, streamText)
	}
}
