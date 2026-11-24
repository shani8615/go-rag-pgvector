// Package rag implements retrieval-augmented question answering over a set
// of ingested documents: chunk -> embed -> store -> retrieve -> answer with
// citations.
package rag

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// NoAnswer is returned verbatim when nothing relevant is found. Deciding
// this in code (rather than hoping the model says it) keeps the bot from
// answering out of thin air.
const NoAnswer = "I don't know. I couldn't find this in the provided documents."

// Embedder turns texts into vectors. Implementations must return one vector
// per input, in order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// ChatModel produces a completion from a system and a user prompt.
type ChatModel interface {
	Complete(ctx context.Context, system, user string) (string, error)
}

// Store persists chunks with their embeddings and runs similarity search.
type Store interface {
	AddDocument(ctx context.Context, source string, chunks []string, vectors [][]float32) (docID string, err error)
	Search(ctx context.Context, vector []float32, k int) ([]Match, error)
}

// Match is a retrieved chunk with its cosine similarity (1 = identical).
type Match struct {
	DocID  string  `json:"doc_id"`
	Source string  `json:"source"`
	Index  int     `json:"chunk_index"`
	Text   string  `json:"text"`
	Score  float64 `json:"score"`
}

// Answer is what Ask returns to callers.
type Answer struct {
	Text    string  `json:"answer"`
	Sources []Match `json:"sources"`
}

// Config tunes retrieval.
type Config struct {
	TopK     int          // chunks to retrieve
	MinScore float64      // drop matches below this similarity
	Chunking ChunkOptions // how documents are split
}

// DefaultConfig is a sensible starting point for text-embedding-3-small.
var DefaultConfig = Config{TopK: 5, MinScore: 0.30, Chunking: DefaultChunkOptions}

// Service wires the pieces together.
type Service struct {
	emb   Embedder
	chat  ChatModel
	store Store
	cfg   Config
}

// ErrEmptyInput is returned for blank documents or questions.
var ErrEmptyInput = errors.New("input is empty")

// NewService builds a Service. Zero-value fields in cfg fall back to defaults.
func NewService(emb Embedder, chat ChatModel, store Store, cfg Config) *Service {
	if cfg.TopK <= 0 {
		cfg.TopK = DefaultConfig.TopK
	}
	if cfg.Chunking.MaxChars <= 0 {
		cfg.Chunking = DefaultConfig.Chunking
	}
	return &Service{emb: emb, chat: chat, store: store, cfg: cfg}
}

// Ingest chunks, embeds and stores a document. It returns the document ID
// and the number of chunks stored.
func (s *Service) Ingest(ctx context.Context, source, text string) (string, int, error) {
	chunks := Chunk(text, s.cfg.Chunking)
	if len(chunks) == 0 {
		return "", 0, ErrEmptyInput
	}
	vecs, err := s.emb.Embed(ctx, chunks)
	if err != nil {
		return "", 0, fmt.Errorf("embed chunks: %w", err)
	}
	if len(vecs) != len(chunks) {
		return "", 0, fmt.Errorf("embedder returned %d vectors for %d chunks", len(vecs), len(chunks))
	}
	id, err := s.store.AddDocument(ctx, source, chunks, vecs)
	if err != nil {
		return "", 0, fmt.Errorf("store document: %w", err)
	}
	return id, len(chunks), nil
}

const systemPrompt = `You answer questions using ONLY the numbered context passages provided.
Rules:
- Cite every claim with the passage number in square brackets, e.g. [1] or [2][3].
- If the passages do not contain the answer, reply exactly: ` + NoAnswer + `
- Do not use outside knowledge. Be concise.`

// Ask retrieves relevant chunks and asks the model to answer from them only.
func (s *Service) Ask(ctx context.Context, question string) (Answer, error) {
	question = strings.TrimSpace(question)
	if question == "" {
		return Answer{}, ErrEmptyInput
	}

	vecs, err := s.emb.Embed(ctx, []string{question})
	if err != nil {
		return Answer{}, fmt.Errorf("embed question: %w", err)
	}
	if len(vecs) != 1 {
		return Answer{}, fmt.Errorf("embedder returned %d vectors for 1 question", len(vecs))
	}

	matches, err := s.store.Search(ctx, vecs[0], s.cfg.TopK)
	if err != nil {
		return Answer{}, fmt.Errorf("search: %w", err)
	}
	relevant := matches[:0:0]
	for _, m := range matches {
		if m.Score >= s.cfg.MinScore {
			relevant = append(relevant, m)
		}
	}
	// Guardrail in code: no relevant context means no model call at all.
	if len(relevant) == 0 {
		return Answer{Text: NoAnswer, Sources: []Match{}}, nil
	}

	out, err := s.chat.Complete(ctx, systemPrompt, buildUserPrompt(question, relevant))
	if err != nil {
		return Answer{}, fmt.Errorf("complete: %w", err)
	}
	out = strings.TrimSpace(out)
	if out == "" || strings.HasPrefix(out, "I don't know") {
		return Answer{Text: NoAnswer, Sources: []Match{}}, nil
	}

	cited := citedSources(out, relevant)
	if len(cited) == 0 {
		// An uncited answer can't be traced back to the documents; refuse it
		// rather than present it as grounded.
		return Answer{Text: NoAnswer, Sources: []Match{}}, nil
	}
	return Answer{Text: out, Sources: cited}, nil
}

func buildUserPrompt(question string, ctxs []Match) string {
	var b strings.Builder
	b.WriteString("Context passages:\n\n")
	for i, m := range ctxs {
		fmt.Fprintf(&b, "[%d] (source: %s)\n%s\n\n", i+1, m.Source, m.Text)
	}
	b.WriteString("Question: ")
	b.WriteString(question)
	return b.String()
}

var citeRe = regexp.MustCompile(`\[(\d+)\]`)

// citedSources returns the passages the answer actually cites, in citation
// order, ignoring out-of-range numbers the model may hallucinate.
func citedSources(answer string, ctxs []Match) []Match {
	seen := map[int]bool{}
	var idx []int
	for _, m := range citeRe.FindAllStringSubmatch(answer, -1) {
		n, err := strconv.Atoi(m[1])
		if err != nil || n < 1 || n > len(ctxs) || seen[n] {
			continue
		}
		seen[n] = true
		idx = append(idx, n)
	}
	sort.Ints(idx)
	out := make([]Match, 0, len(idx))
	for _, n := range idx {
		out = append(out, ctxs[n-1])
	}
	return out
}
