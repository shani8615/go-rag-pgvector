package rag

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// HashEmbedder is a deterministic, offline bag-of-words embedder. It is NOT a
// semantic model; it exists so tests and the offline demo mode run without
// an API key. Texts that share words get similar vectors.
type HashEmbedder struct{ Dim int }

// Embed implements Embedder.
func (h HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	dim := h.Dim
	if dim <= 0 {
		dim = 256
	}
	out := make([][]float32, len(texts))
	for i, t := range texts {
		v := make([]float32, dim)
		for _, w := range tokens(t) {
			f := fnv.New32a()
			_, _ = f.Write([]byte(w))
			v[f.Sum32()%uint32(dim)]++
		}
		normalizeVec(v)
		out[i] = v
	}
	return out, nil
}

var stopwords = map[string]bool{
	"the": true, "a": true, "an": true, "and": true, "or": true, "of": true,
	"to": true, "in": true, "is": true, "are": true, "for": true, "on": true,
	"what": true, "how": true, "do": true, "does": true, "i": true, "you": true,
	"it": true, "with": true, "be": true, "can": true, "my": true, "our": true,
}

func tokens(s string) []string {
	f := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := f[:0]
	for _, w := range f {
		if len(w) > 1 && !stopwords[w] {
			out = append(out, w)
		}
	}
	return out
}

func normalizeVec(v []float32) {
	var n float64
	for _, x := range v {
		n += float64(x) * float64(x)
	}
	if n == 0 {
		return
	}
	n = math.Sqrt(n)
	for i := range v {
		v[i] = float32(float64(v[i]) / n)
	}
}

// ExtractiveModel is an offline ChatModel that "answers" by quoting the line
// of the top context passage that shares the most words with the question.
// Like HashEmbedder, it is for tests and the offline demo only.
type ExtractiveModel struct{}

// Complete implements ChatModel.
func (ExtractiveModel) Complete(_ context.Context, _ string, user string) (string, error) {
	start := strings.Index(user, "[1] (source: ")
	qi := strings.LastIndex(user, "\nQuestion: ")
	if start < 0 || qi < start {
		return NoAnswer, nil
	}
	question := user[qi+len("\nQuestion: "):]
	body := user[start:qi]
	if nl := strings.Index(body, "\n"); nl >= 0 {
		body = body[nl+1:]
	}
	if end := strings.Index(body, "\n\n[2] (source: "); end >= 0 {
		body = body[:end]
	}

	want := map[string]bool{}
	for _, w := range tokens(question) {
		want[w] = true
	}
	best, bestScore := "", 0
	for _, line := range strings.FieldsFunc(body, func(r rune) bool { return r == '\n' || r == '.' }) {
		line = strings.TrimSpace(strings.TrimLeft(line, "#-* "))
		score := 0
		for _, w := range tokens(line) {
			if want[w] {
				score++
			}
		}
		if score > bestScore {
			best, bestScore = line, score
		}
	}
	if best == "" {
		return NoAnswer, nil
	}
	return best + ". [1]", nil
}
