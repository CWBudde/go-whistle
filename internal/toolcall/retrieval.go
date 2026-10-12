package toolcall

import (
	"errors"
	"fmt"
)

// RetrievalTools is the number of tools a turn renders when the engine
// retrieves: upstream documents that with more tools a built-in embedding
// head scores every schema against the query and only the best
// RetrievalTools enter the prompt (docs/FINDINGS.md F-049).
const RetrievalTools = 5

// Retrieves reports whether the engine shortlists the tools per turn: only
// a model with an embedding head (heads.manifest code 1,
// cact.HeadEmbedding) and more than RetrievalTools tools. Otherwise every
// declared tool is rendered into the static prefix. The pinned needle3.cact
// carries only the confidence head, and engine 3.2.0 renders all of 1 to
// 239 tools and writes no tool index; the branch with an embedding head is
// upstream's description, not measured.
func Retrieves(embeddingHead bool, tools int) bool {
	return embeddingHead && tools > RetrievalTools
}

// ErrContextBudget is needle_init's refusal of a static prefix that does
// not fit the model's context.
var ErrContextBudget = errors.New("toolcall: context budget exceeded")

// CheckPrefix is needle_init's budget check: a static prefix of context
// tokens or more (the model's max_position_embeddings, 8192 for needle3)
// is refused, one token less is accepted (docs/FINDINGS.md F-049).
func CheckPrefix(prefix []int, context int) error {
	if len(prefix) >= context {
		return fmt.Errorf("%w: static prefix is %d tokens, model context is %d tokens",
			ErrContextBudget, len(prefix), context)
	}

	return nil
}
