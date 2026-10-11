package text

import (
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
	"github.com/cwbudde/go-whistle/internal/weights"
)

func needle3Config() Config {
	return Config{
		Vocab: 8192, OutVocab: 8192, Dim: 768, HadaN: 1024, Heads: 12, KVHeads: 2, QKDim: 48, VDim: 64,
		Layers: 20, MaxSeqLen: 8192, Lanes: 4, Taps: 3, SlidingWindow: 1024, GlobalLayers: []int{4, 9, 14, 19},
		KVWindow: 256, KVBits: 8, Sites: []int{3, 7, 11, 15, 19}, Orders: []int{2, 3},
		Slots: 18432, Tables: 6, SubDim: 128, ConvTaps: 4, Rope: 100000,
	}
}

// TestConfigHeaderRoundTrip maps a config to the header export.py writes
// and back.
func TestConfigHeaderRoundTrip(t *testing.T) {
	cfg := needle3Config()

	h, err := cfg.Header()
	if err != nil {
		t.Fatal(err)
	}

	if h.GlobalMaskLo != 1<<4|1<<9|1<<14|1<<19 || h.EngramConvDilation != 3 || h.NumEngramSites != 5 {
		t.Errorf("Header() = %+v", h)
	}

	if got := ConfigFromHeader(h); !reflect.DeepEqual(got, cfg) {
		t.Errorf("ConfigFromHeader(Header()) =\n%+v\nwant\n%+v", got, cfg)
	}

	cfg.Layers = 65

	_, err = cfg.Header()
	if err == nil {
		t.Error("Header accepted 65 layers")
	}
}

// TestConfigBothFormats derives the geometry from needle3.cact's header and
// from needle3.safetensors' config: equal, and the header a text export of
// the safetensors config would write is the file's (docs/FINDINGS.md
// F-045).
func TestConfigBothFormats(t *testing.T) {
	requireFile(t, needle3Cact)
	requireFile(t, needle3Safetensors)

	c, err := Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	s, err := Open(needle3Safetensors)
	if err != nil {
		t.Fatal(err)
	}

	if want := needle3Config(); !reflect.DeepEqual(c.TextConfig(), want) || !reflect.DeepEqual(s.TextConfig(), want) {
		t.Errorf("configs:\n.cact        %+v\nsafetensors  %+v\nwant         %+v", c.TextConfig(), s.TextConfig(), want)
	}

	h, err := s.TextConfig().Header()
	if err != nil {
		t.Fatal(err)
	}

	f, err := cact.Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	want := f.Header()
	want.Tag, want.NumTensors, want.CodebookLen = 0, 0, 0

	if h != want {
		t.Errorf("Header() from safetensors =\n%+v\nneedle3.cact\n%+v", h, want)
	}
}

// TestSourceBothFormats serves every name from both formats: the same names
// in the same order, the same shapes, the derived records and the FP16
// records bit for bit (the checkpoint is F16), and every CQ record within
// the cosine docs/FINDINGS.md F-045 measured. The tokenizer exists only in
// the .cact.
func TestSourceBothFormats(t *testing.T) {
	requireFile(t, needle3Cact)
	requireFile(t, needle3Safetensors)

	c, err := Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	s, err := Open(needle3Safetensors)
	if err != nil {
		t.Fatal(err)
	}

	if !slices.Equal(c.Names(), s.Names()) {
		t.Fatalf("names differ: %d in the .cact, %d from safetensors", len(c.Names()), len(s.Names()))
	}

	f, err := cact.Open(needle3Cact)
	if err != nil {
		t.Fatal(err)
	}

	var exact, cq int

	for i, name := range c.Names() {
		cs, _ := c.Shape(name)
		ss, _ := s.Shape(name)

		if !slices.Equal(cs, ss) {
			t.Errorf("%s: shapes %v and %v", name, cs, ss)
		}

		if name == "tokenizer" {
			_, err := s.Raw(name)
			if !errors.Is(err, weights.ErrUnavailable) {
				t.Errorf("safetensors tokenizer: %v, want ErrUnavailable", err)
			}

			blob, err := c.Raw(name)
			if err != nil || len(blob) != 109140 {
				t.Errorf(".cact tokenizer: %d bytes, %v", len(blob), err)
			}

			continue
		}

		cv, _, err := c.Tensor(name)
		if err != nil {
			t.Fatalf(".cact %s: %v", name, err)
		}

		sv, _, err := s.Tensor(name)
		if err != nil {
			t.Fatalf("safetensors %s: %v", name, err)
		}

		rec, _ := f.Record(i)
		if rec.DType == cact.CQ {
			cq++

			if c := cosine(cv, sv); c < 0.9333 {
				t.Errorf("%s: CQ%d cosine %.6f", name, rec.Bits, c)
			}
		} else {
			exact++

			if !slices.Equal(cv, sv) {
				t.Errorf("%s: %s values differ", name, rec.DType)
			}
		}

		s.Release(name)
		c.Release(name)
	}

	t.Logf("%d records equal bit for bit, %d CQ records within cosine, 1 tokenizer", exact, cq)

	err = c.Close()
	if err != nil {
		t.Error(err)
	}

	_, _, err = c.Tensor("final_norm")
	if !errors.Is(err, weights.ErrClosed) {
		t.Errorf("Tensor after Close: %v", err)
	}
}
