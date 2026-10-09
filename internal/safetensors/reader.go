// Package safetensors reads and writes safetensors files: an 8-byte LE header
// length, a JSON header (tensor entries plus an optional string map under
// __metadata__), then the raw tensor data.
//
// Copied from github.com/cwbudde/go-pocket-tts internal/safetensors @ 9faba38,
// without the pocket-tts voice and Mimi helpers; Metadata, Config, Run and
// Step are go-whistle additions. Port fixes in either direction.
package safetensors

import "errors"

// Tensor holds a single tensor loaded from a safetensors file. Data is
// float32 whatever the file stores; DType names the file's dtype (Store.Tensor
// sets it, EncodeTensors writes F32 when it is empty).
type Tensor struct {
	Name  string
	DType string
	Shape []int64
	Data  []float32
}

// LoadFirstTensor reads a safetensors file and returns the first float32 tensor.
func LoadFirstTensor(path string) (*Tensor, error) {
	store, err := OpenStore(path, StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	names := store.Names()
	if len(names) == 0 {
		return nil, errors.New("safetensors: no tensors found")
	}

	return store.Tensor(names[0])
}

// LoadFirstTensorFromBytes decodes a safetensors payload and returns the first
// float32 tensor.
func LoadFirstTensorFromBytes(data []byte) (*Tensor, error) {
	store, err := OpenStoreFromBytes(data, StoreOptions{})
	if err != nil {
		return nil, err
	}
	defer store.Close()

	names := store.Names()
	if len(names) == 0 {
		return nil, errors.New("safetensors: no tensors found")
	}

	return store.Tensor(names[0])
}
