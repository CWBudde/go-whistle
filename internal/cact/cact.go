package cact

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"math/bits"
	"os"
	"slices"
)

// ErrDType reports a reader called on a tensor of another dtype.
var ErrDType = errors.New("wrong dtype")

// File is a parsed .cact container. Parse validates the whole directory up
// front, so the readers only check the index and the dtype.
type File struct {
	data     []byte
	header   Header
	codebook []float32
	records  []Record
}

// Open reads and parses the file at path.
func Open(path string) (*File, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cact: %w", err)
	}

	return Parse(data)
}

// Parse validates and indexes data, which the File keeps referencing.
func Parse(data []byte) (*File, error) {
	if len(data) < HeaderSize {
		return nil, fmt.Errorf("cact: %d bytes is shorter than the %d-byte header", len(data), HeaderSize)
	}

	var h Header

	_, err := binary.Decode(data[:HeaderSize], binary.LittleEndian, &h)
	if err != nil {
		return nil, fmt.Errorf("cact: decode header: %w", err)
	}

	err = checkHeader(h)
	if err != nil {
		return nil, err
	}

	size := uint64(len(data))
	dirStart := uint64(HeaderSize + 4*CodebookLen)
	dirEnd := dirStart + uint64(h.NumTensors)*RecordSize

	if dirEnd > size {
		return nil, fmt.Errorf("cact: directory of %d records ends at %d, past EOF (%d bytes)", h.NumTensors, dirEnd, size)
	}

	codebook := make([]float32, CodebookLen)
	for i := range codebook {
		codebook[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[HeaderSize+4*i:]))
	}

	records := make([]Record, h.NumTensors)
	prevEnd := dirEnd

	for i := range records {
		at := dirStart + uint64(i)*RecordSize

		_, err := binary.Decode(data[at:at+RecordSize], binary.LittleEndian, &records[i])
		if err != nil {
			return nil, fmt.Errorf("cact: decode record %d: %w", i, err)
		}

		err = checkRecord(records[i], prevEnd, size)
		if err != nil {
			return nil, fmt.Errorf("cact: tensor %d (%s): %w", i, records[i].DType, err)
		}

		prevEnd = records[i].Offset + records[i].NBytes
	}

	return &File{data: data, header: h, codebook: codebook, records: records}, nil
}

func checkHeader(h Header) error {
	switch {
	case h.Tag != Tag:
		return fmt.Errorf("cact: bad tag 0x%08x, want 0x%08x", h.Tag, Tag)
	case h.CodebookLen != CodebookLen:
		return fmt.Errorf("cact: codebook length %d, want %d", h.CodebookLen, CodebookLen)
	case h.NumEngramOrders > uint32(len(h.EngramOrders)):
		return fmt.Errorf("cact: %d engram orders, the header holds %d", h.NumEngramOrders, len(h.EngramOrders))
	case h.NumEngramSites > uint32(len(h.EngramSites)):
		return fmt.Errorf("cact: %d engram sites, the header holds %d", h.NumEngramSites, len(h.EngramSites))
	}

	return nil
}

// checkRecord validates one directory entry against the file size and the
// end of the previous blob (blobs are laid out in directory order).
func checkRecord(r Record, prevEnd, size uint64) error {
	if r.Pad != 0 {
		return fmt.Errorf("non-zero pad %d", r.Pad)
	}

	if r.NDim > MaxDims {
		return fmt.Errorf("ndim %d > %d", r.NDim, MaxDims)
	}

	for k := int(r.NDim); k < MaxDims; k++ {
		if r.Shape[k] != 0 {
			return fmt.Errorf("shape %v has a dimension past ndim %d", r.Shape, r.NDim)
		}
	}

	want, err := blobSize(r)
	if err != nil {
		return err
	}

	switch {
	case r.Offset%Align != 0:
		return fmt.Errorf("offset %d is not %d-byte aligned", r.Offset, Align)
	case r.Offset < prevEnd:
		return fmt.Errorf("blob at %d overlaps the directory or the previous blob (ends at %d)", r.Offset, prevEnd)
	case r.Offset > size || r.NBytes > size-r.Offset:
		return fmt.Errorf("blob [%d, +%d) runs past EOF (%d bytes)", r.Offset, r.NBytes, size)
	case want >= 0 && uint64(want) != r.NBytes:
		return fmt.Errorf("nbytes %d, want %d for shape %v", r.NBytes, want, r.Dims())
	}

	return nil
}

// blobSize is the byte count the record's dtype and shape imply, or -1 for
// RAW, whose length is free.
func blobSize(r Record) (int64, error) {
	switch r.DType {
	case FP16, FP32:
		numel := uint64(1)
		for _, d := range r.Shape[:r.NDim] {
			numel = mulChecked(numel, uint64(d))
		}

		elem := uint64(2)
		if r.DType == FP32 {
			elem = 4
		}

		return toSize(mulChecked(numel, elem))
	case RAW:
		if r.NDim != 0 {
			return 0, fmt.Errorf("RAW needs ndim 0, got %d", r.NDim)
		}

		return -1, nil
	case CQ:
		return cqBlobSize(r)
	default:
		return 0, fmt.Errorf("unknown %s", r.DType)
	}
}

// cqBlobSize follows export.py _packed_row_bytes: per row the packed indices
// of the group-padded input, then one FP16 norm per group.
func cqBlobSize(r Record) (int64, error) {
	if r.NDim != 2 {
		return 0, fmt.Errorf("CQ needs ndim 2, got %d", r.NDim)
	}

	if r.Group == 0 || r.Group%8 != 0 {
		return 0, fmt.Errorf("CQ group %d is not a positive multiple of 8", r.Group)
	}

	if r.Bits < 1 || r.Bits > 5 {
		return 0, fmt.Errorf("CQ bits %d not in 1..5", r.Bits)
	}

	out, in, group := uint64(r.Shape[0]), uint64(r.Shape[1]), uint64(r.Group)
	groups := (in + group - 1) / group
	inPad := groups * group

	idxBits := uint64(r.Bits)
	if r.Bits == 5 { // ternary: four 2-bit crumbs per byte
		idxBits = 2
	}

	rowBytes := inPad*idxBits/8 + groups*2

	return toSize(mulChecked(out, rowBytes))
}

// mulChecked multiplies, saturating to MaxUint64 on overflow so toSize
// rejects it.
func mulChecked(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	if hi != 0 {
		return math.MaxUint64
	}

	return lo
}

func toSize(n uint64) (int64, error) {
	if n > math.MaxInt64 {
		return 0, errors.New("blob size overflows")
	}

	return int64(n), nil
}

// Header returns the file header.
func (f *File) Header() Header { return f.header }

// Len is the number of tensors.
func (f *File) Len() int { return len(f.records) }

// Record returns directory entry i.
func (f *File) Record(i int) (Record, error) {
	if i < 0 || i >= len(f.records) {
		return Record{}, fmt.Errorf("cact: tensor %d out of range [0, %d)", i, len(f.records))
	}

	return f.records[i], nil
}

// Codebook returns a copy of the stored unit codebook for a CQ width of 2,
// 3 or 4 bits. The 1-bit and ternary codebooks are analytic and not stored.
func (f *File) Codebook(width int) ([]float32, error) {
	var from, to int

	switch width {
	case 2:
		from, to = 0, 4
	case 3:
		from, to = 4, 12
	case 4:
		from, to = 12, 28
	default:
		return nil, fmt.Errorf("cact: no stored codebook for %d-bit CQ", width)
	}

	return slices.Clone(f.codebook[from:to]), nil
}

// FP32 decodes an FP32 tensor.
func (f *File) FP32(i int) ([]float32, error) {
	blob, err := f.blob(i, FP32)
	if err != nil {
		return nil, err
	}

	out := make([]float32, len(blob)/4)
	for k := range out {
		out[k] = math.Float32frombits(binary.LittleEndian.Uint32(blob[4*k:]))
	}

	return out, nil
}

// FP16 decodes an FP16 tensor, widened to float32.
func (f *File) FP16(i int) ([]float32, error) {
	blob, err := f.blob(i, FP16)
	if err != nil {
		return nil, err
	}

	out := make([]float32, len(blob)/2)
	for k := range out {
		out[k] = float16ToFloat32(binary.LittleEndian.Uint16(blob[2*k:]))
	}

	return out, nil
}

// Float32 decodes an FP16 or FP32 tensor.
func (f *File) Float32(i int) ([]float32, error) {
	rec, err := f.Record(i)
	if err != nil {
		return nil, err
	}

	if rec.DType == FP16 {
		return f.FP16(i)
	}

	return f.FP32(i)
}

// Raw returns a copy of a RAW tensor's bytes.
func (f *File) Raw(i int) ([]byte, error) {
	blob, err := f.blob(i, RAW)
	if err != nil {
		return nil, err
	}

	return slices.Clone(blob), nil
}

func (f *File) blob(i int, want DType) ([]byte, error) {
	rec, err := f.Record(i)
	if err != nil {
		return nil, err
	}

	if rec.DType != want {
		return nil, fmt.Errorf("cact: tensor %d is %s, not %s: %w", i, rec.DType, want, ErrDType)
	}

	return f.data[rec.Offset : rec.Offset+rec.NBytes], nil
}
