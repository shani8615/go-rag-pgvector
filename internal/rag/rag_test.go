package rag

import (
	"context"
	"errors"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestChunkEmpty(t *testing.T) {
	if got := Chunk("  \n\n \r\n ", DefaultChunkOptions); got != nil {
		t.Fatalf("want no chunks, got %q", got)
	}
}

func TestChunkKeepsShortDocWhole(t *testing.T) {
	got := Chunk("Hello world.\n\nSecond paragraph.", DefaultChunkOptions)
	if len(got) != 1 || !strings.Contains(got[0], "Second paragraph.") {
		t.Fatalf("got %q", got)
	}
}

func TestChunkRespectsMaxAndOverlap(t *testing.T) {
	para := strings.Repeat("Go services stay simple when each one does one thing well. ", 10)
	doc := strings.Join([]string{para, para, para, para}, "\n\n")
	opt := ChunkOptions{MaxChars: 400, Overlap: 60}

	got := Chunk(doc, opt)
	if len(got) < 4 {
		t.Fatalf("expected several chunks, got %d", len(got))
	}
	for i, c := range got {
		// A chunk may exceed MaxChars only by the overlap carried in.
		if len(c) > opt.MaxChars+opt.Overlap+2 {
			t.Errorf("chunk %d too long: %d chars", i, len(c))
		}
		if !utf8.ValidString(c) {
			t.Errorf("chunk %d is not valid UTF-8", i)
		}
	}
}

func TestChunkNeverSplitsRunes(t *testing.T) {
	doc := strings.Repeat("کراچی", 500) // no spaces or sentence breaks
	for i, c := range Chunk(doc, ChunkOptions{MaxChars: 101, Overlap: 17}) {
		if !utf8.ValidString(c) {
			t.Fatalf("chunk %d is not valid UTF-8", i)
		}
	}
}

func TestCosine(t *testing.T) {
	if got := Cosine([]float32{1, 0}, []float32{1, 0}); got < 0.999 {
		t.Errorf("identical vectors: got %v", got)
	}
	if got := Cosine([]float32{1, 0}, []float32{0, 1}); got != 0 {
		t.Errorf("orthogonal vectors: got %v", got)
	}
	if got := Cosine([]float32{1}, []float32{1, 2}); got != 0 {
		t.Errorf("dimension mismatch should be 0, got %v", got)
	}
}

const policyDoc = `Refund policy

Customers can request a full refund within 30 days of purchase.
Refunds are paid back to the original payment method within 5 business days.

Shipping

Orders ship from our Karachi warehouse within 2 business days.`

func newTestService(chat ChatModel) *Service {
	return NewService(HashEmbedder{Dim: 512}, chat, NewMemStore(), DefaultConfig)
}

func TestAskAnswersWithCitation(t *testing.T) {
	ctx := context.Background()
	s := newTestService(ExtractiveModel{})
	if _, n, err := s.Ingest(ctx, "policy.md", policyDoc); err != nil || n == 0 {
		t.Fatalf("ingest: n=%d err=%v", n, err)
	}

	ans, err := s.Ask(ctx, "How many days do customers have to request a refund?")
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(ans.Text, "30 days") {
		t.Errorf("answer should contain the fact, got %q", ans.Text)
	}
	if len(ans.Sources) == 0 || ans.Sources[0].Source != "policy.md" {
		t.Errorf("want policy.md as source, got %+v", ans.Sources)
	}
}

// countingModel records whether the model was called at all.
type countingModel struct {
	calls int
	reply string
}

func (m *countingModel) Complete(context.Context, string, string) (string, error) {
	m.calls++
	return m.reply, nil
}

func TestAskRefusesWithoutRelevantContextAndSkipsModel(t *testing.T) {
	ctx := context.Background()
	m := &countingModel{reply: "Made-up answer [1]"}
	s := newTestService(m)
	if _, _, err := s.Ingest(ctx, "policy.md", policyDoc); err != nil {
		t.Fatal(err)
	}

	ans, err := s.Ask(ctx, "Who won the football world cup?")
	if err != nil {
		t.Fatal(err)
	}
	if ans.Text != NoAnswer {
		t.Errorf("want NoAnswer, got %q", ans.Text)
	}
	if m.calls != 0 {
		t.Errorf("model must not be called without relevant context, got %d calls", m.calls)
	}
}

func TestAskRejectsUncitedAnswer(t *testing.T) {
	ctx := context.Background()
	s := newTestService(&countingModel{reply: "Refunds are always instant."})
	if _, _, err := s.Ingest(ctx, "policy.md", policyDoc); err != nil {
		t.Fatal(err)
	}
	ans, err := s.Ask(ctx, "refund policy days")
	if err != nil {
		t.Fatal(err)
	}
	if ans.Text != NoAnswer {
		t.Errorf("uncited answer should be refused, got %q", ans.Text)
	}
}

func TestCitedSourcesIgnoresOutOfRange(t *testing.T) {
	ctxs := []Match{{Source: "a"}, {Source: "b"}}
	got := citedSources("x [2] y [7] z [2] [1]", ctxs)
	if len(got) != 2 || got[0].Source != "a" || got[1].Source != "b" {
		t.Fatalf("got %+v", got)
	}
}

func TestEmptyInputs(t *testing.T) {
	ctx := context.Background()
	s := newTestService(ExtractiveModel{})
	if _, _, err := s.Ingest(ctx, "x", "   "); !errors.Is(err, ErrEmptyInput) {
		t.Errorf("ingest: want ErrEmptyInput, got %v", err)
	}
	if _, err := s.Ask(ctx, " "); !errors.Is(err, ErrEmptyInput) {
		t.Errorf("ask: want ErrEmptyInput, got %v", err)
	}
}
