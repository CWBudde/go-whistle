package text

import "regexp"

// weightGroup is the memory one group of tensors holds (docs/FINDINGS.md
// F-059): the packed matrices' indices (aliasing the file data), float32
// norms and codebooks, and lut copies once built (cq.Matrix.Bytes), and
// the float32 tensors in the cache.
type weightGroup struct {
	group                    string
	index, norms, lut, float int
}

func (g weightGroup) total() int { return g.index + g.norms + g.lut + g.float }

// groupPattern puts a name in a weightGroup.
type groupPattern struct {
	group string
	re    *regexp.Regexp
}

// weightGroups are the groups weightBytes reports, in order; a name falls
// in the first whose pattern it matches.
var weightGroups = []groupPattern{
	{"embedding", regexp.MustCompile(`^embedding$`)},
	{"blocks", regexp.MustCompile(`^layer\d+\.`)},
	{"mHC", regexp.MustCompile(`^mhc_`)},
	{"engram tables", regexp.MustCompile(`^engram\d+\.tables$`)},
	{"engram", regexp.MustCompile(`^engram\d+\.`)},
	{"heads", regexp.MustCompile(`_head\.`)},
	{"other", regexp.MustCompile(``)},
}

func groupOf(name string) int {
	for k, g := range weightGroups {
		if g.re.MatchString(name) {
			return k
		}
	}

	return len(weightGroups) - 1
}

// weightBytes is the memory the source holds for the model now, by group:
// every packed matrix built and every float32 tensor cached.
func (s *Source) weightBytes() []weightGroup {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]weightGroup, len(weightGroups))
	for k, g := range weightGroups {
		out[k].group = g.group
	}

	for i, e := range s.names {
		g := &out[groupOf(e.Name)]

		if v, ok := s.cache[e.Name]; ok {
			g.float += 4 * len(v)
		}

		m, ok := s.mats[e.Name]
		if !ok {
			continue
		}

		// cq.NewGather keeps the index as stored and widens the FP16
		// norms to float32.
		p, err := s.file.CQPacked(i)
		if err != nil {
			continue
		}

		index, norms := len(p.Index), 2*len(p.Norms)+4*len(p.Codebook)
		g.index += index
		g.norms += norms
		g.lut += m.Bytes() - index - norms
	}

	return out
}

// cached lists, in .cact order, the names the float32 cache holds.
func (s *Source) cached() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	var names []string

	for _, e := range s.names {
		if _, ok := s.cache[e.Name]; ok {
			names = append(names, e.Name)
		}
	}

	return names
}

// packedNames lists, in .cact order, the names Matrix has built a packed
// matrix for.
func (s *Source) packedNames() []string {
	s.mu.Lock()
	defer s.mu.Unlock()

	var names []string

	for _, e := range s.names {
		if _, ok := s.mats[e.Name]; ok {
			names = append(names, e.Name)
		}
	}

	return names
}
