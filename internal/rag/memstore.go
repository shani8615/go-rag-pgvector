package rag

import (
	"context"
	"fmt"
	"math"
	"sort"
	"sync"
)

// MemStore is an in-memory Store using brute-force cosine similarity. It is
// used in tests and for running the service without a database.
type MemStore struct {
	mu     sync.RWMutex
	nextID int
	rows   []memRow
}

type memRow struct {
	m   Match
	vec []float32
}

// NewMemStore returns an empty in-memory store.
func NewMemStore() *MemStore { return &MemStore{} }

// AddDocument implements Store.
func (s *MemStore) AddDocument(_ context.Context, source string, chunks []string, vectors [][]float32) (string, error) {
	if len(chunks) != len(vectors) {
		return "", fmt.Errorf("got %d chunks and %d vectors", len(chunks), len(vectors))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.nextID++
	id := fmt.Sprintf("mem-%d", s.nextID)
	for i, c := range chunks {
		s.rows = append(s.rows, memRow{
			m:   Match{DocID: id, Source: source, Index: i, Text: c},
			vec: vectors[i],
		})
	}
	return id, nil
}

// Search implements Store.
func (s *MemStore) Search(_ context.Context, vector []float32, k int) ([]Match, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Match, 0, len(s.rows))
	for _, r := range s.rows {
		m := r.m
		m.Score = Cosine(vector, r.vec)
		out = append(out, m)
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Score > out[j].Score })
	if k > 0 && len(out) > k {
		out = out[:k]
	}
	return out, nil
}

// Cosine returns the cosine similarity of a and b, or 0 if either is empty,
// zero-length or they differ in dimension.
func Cosine(a, b []float32) float64 {
	if len(a) == 0 || len(a) != len(b) {
		return 0
	}
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
