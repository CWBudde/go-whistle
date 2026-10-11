package whistle

import (
	"encoding/binary"
	"os"
	"slices"
	"strings"
	"testing"

	"github.com/cwbudde/go-whistle/internal/cact"
)

// LoadBytes on the .cact bytes is the model Load opens from the file: the
// same transcript, language and words on a short clip.
func TestLoadBytes(t *testing.T) {
	m := loadModel(t)

	data, err := os.ReadFile(cactPath)
	if err != nil {
		t.Fatal(err)
	}

	b, err := LoadBytes(data, testOptions(t)...)
	if err != nil {
		t.Fatal(err)
	}

	defer b.Close()

	c := loadClips(t, shortClips[0])[0]
	opts := TranscribeOptions{WordTimestamps: true}

	want, err := m.Transcribe(c.Samples, opts)
	if err != nil {
		t.Fatal(err)
	}

	got, err := b.Transcribe(c.Samples, opts)
	if err != nil {
		t.Fatal(err)
	}

	if got.Text != want.Text || got.Language != want.Language || !slices.Equal(got.Words, want.Words) {
		t.Errorf("LoadBytes: [%s] %q %v, Load [%s] %q %v", got.Language, got.Text, got.Words, want.Language, want.Text, want.Words)
	}

	if !int8Mode(t) && (got.Text != c.Text || got.Language != c.Language) {
		t.Errorf("LoadBytes: [%s] %q, the engine [%s] %q", got.Language, got.Text, c.Language, c.Text)
	}
}

// A safetensors checkpoint holds no tokenizer, and LoadBytes has no file
// to find whistle.cact next to: it needs WithVocabulary.
func TestLoadBytesSafetensorsNeedsVocabulary(t *testing.T) {
	requireFile(t, safetensorsPath)
	requireFile(t, cactPath)

	data, err := os.ReadFile(safetensorsPath)
	if err != nil {
		t.Fatal(err)
	}

	_, err = LoadBytes(data)
	if err == nil || !strings.Contains(err.Error(), "WithVocabulary") {
		t.Errorf("LoadBytes without a vocabulary: %v, want an error naming WithVocabulary", err)
	}

	m, err := LoadBytes(data, WithVocabulary(cactPath))
	if err != nil {
		t.Fatalf("LoadBytes with WithVocabulary: %v", err)
	}

	_ = m.Close()
}

func TestLoadBytesRejectsGarbage(t *testing.T) {
	tag := binary.LittleEndian.AppendUint32(nil, cact.Tag)

	for name, data := range map[string][]byte{
		"empty":         nil,
		"text":          []byte("not a weight file"),
		"truncated tag": tag[:2],
		"cact header":   append(tag, 0, 0, 0, 0),
	} {
		m, err := LoadBytes(data)
		if err == nil {
			_ = m.Close()

			t.Errorf("%s: no error", name)
		} else if !strings.HasPrefix(err.Error(), "whistle: ") {
			t.Errorf("%s: %v, want the whistle: prefix", name, err)
		}
	}
}
