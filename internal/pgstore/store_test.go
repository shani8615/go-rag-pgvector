package pgstore

import (
	"context"
	"os"
	"testing"

	"github.com/shani8615/go-rag-pgvector/internal/rag"
)

// Integration test: runs only when TEST_DATABASE_URL points at a Postgres
// with pgvector (CI starts one; locally use `docker compose up -d db`).
func TestStoreRoundTrip(t *testing.T) {
	dsn := os.Getenv("TEST_DATABASE_URL")
	if dsn == "" {
		t.Skip("TEST_DATABASE_URL not set")
	}
	ctx := context.Background()
	s, err := Open(ctx, dsn, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	id, err := s.AddDocument(ctx, "test.md",
		[]string{"about cats", "about dogs"},
		[][]float32{{1, 0, 0}, {0, 1, 0}},
	)
	if err != nil || id == "" {
		t.Fatalf("add: id=%q err=%v", id, err)
	}

	got, err := s.Search(ctx, []float32{0.9, 0.1, 0}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 || got[0].Text != "about cats" || got[0].Source != "test.md" {
		t.Fatalf("got %+v", got)
	}
	if got[0].Score < 0.9 {
		t.Errorf("score = %v, want close to 1", got[0].Score)
	}
}

func TestVectorLiteral(t *testing.T) {
	if got := vectorLiteral([]float32{1, -0.5, 0.25}); got != "[1,-0.5,0.25]" {
		t.Fatalf("got %s", got)
	}
}

var _ rag.Store = (*Store)(nil)
