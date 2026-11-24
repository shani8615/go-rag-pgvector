// Package pgstore is a rag.Store backed by PostgreSQL with the pgvector
// extension, using an HNSW index for cosine-similarity search.
package pgstore

import (
	"context"
	"fmt"
	"strconv"
	"strings"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	"github.com/shani8615/go-rag-pgvector/internal/rag"
)

// Store implements rag.Store.
type Store struct {
	pool *pgxpool.Pool
}

// Open connects to Postgres and creates the schema if needed. dim must match
// the embedding model's output size (1536 for text-embedding-3-small).
func Open(ctx context.Context, dsn string, dim int) (*Store, error) {
	if dim <= 0 || dim > 16000 {
		return nil, fmt.Errorf("invalid embedding dimension %d", dim)
	}
	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return nil, fmt.Errorf("connect: %w", err)
	}
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("ping: %w", err)
	}
	s := &Store{pool: pool}
	if err := s.migrate(ctx, dim); err != nil {
		pool.Close()
		return nil, fmt.Errorf("migrate: %w", err)
	}
	return s, nil
}

// Close releases the connection pool.
func (s *Store) Close() { s.pool.Close() }

func (s *Store) migrate(ctx context.Context, dim int) error {
	stmts := []string{
		`CREATE EXTENSION IF NOT EXISTS vector`,
		`CREATE TABLE IF NOT EXISTS documents (
			id         BIGSERIAL PRIMARY KEY,
			source     TEXT        NOT NULL,
			created_at TIMESTAMPTZ NOT NULL DEFAULT now()
		)`,
		// dim is a validated integer, not user input.
		fmt.Sprintf(`CREATE TABLE IF NOT EXISTS chunks (
			id          BIGSERIAL PRIMARY KEY,
			document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
			chunk_index INT    NOT NULL,
			content     TEXT   NOT NULL,
			embedding   vector(%d) NOT NULL
		)`, dim),
		`CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
			ON chunks USING hnsw (embedding vector_cosine_ops)`,
	}
	for _, q := range stmts {
		if _, err := s.pool.Exec(ctx, q); err != nil {
			return err
		}
	}
	return nil
}

// AddDocument implements rag.Store. The document and all its chunks are
// written in one transaction, so a failed ingest leaves nothing behind.
func (s *Store) AddDocument(ctx context.Context, source string, chunks []string, vectors [][]float32) (string, error) {
	if len(chunks) != len(vectors) {
		return "", fmt.Errorf("got %d chunks and %d vectors", len(chunks), len(vectors))
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return "", err
	}
	defer tx.Rollback(ctx) //nolint:errcheck // no-op after Commit

	var docID int64
	if err := tx.QueryRow(ctx,
		`INSERT INTO documents (source) VALUES ($1) RETURNING id`, source,
	).Scan(&docID); err != nil {
		return "", fmt.Errorf("insert document: %w", err)
	}

	batch := &pgx.Batch{}
	for i := range chunks {
		batch.Queue(
			`INSERT INTO chunks (document_id, chunk_index, content, embedding)
			 VALUES ($1, $2, $3, $4::text::vector)`,
			docID, i, chunks[i], vectorLiteral(vectors[i]),
		)
	}
	br := tx.SendBatch(ctx, batch)
	for range chunks {
		if _, err := br.Exec(); err != nil {
			_ = br.Close()
			return "", fmt.Errorf("insert chunk: %w", err)
		}
	}
	if err := br.Close(); err != nil {
		return "", err
	}
	if err := tx.Commit(ctx); err != nil {
		return "", err
	}
	return strconv.FormatInt(docID, 10), nil
}

// Search implements rag.Store using pgvector's cosine distance operator.
func (s *Store) Search(ctx context.Context, vector []float32, k int) ([]rag.Match, error) {
	rows, err := s.pool.Query(ctx, `
		SELECT c.document_id, d.source, c.chunk_index, c.content,
		       1 - (c.embedding <=> $1::text::vector) AS score
		FROM chunks c
		JOIN documents d ON d.id = c.document_id
		ORDER BY c.embedding <=> $1::text::vector
		LIMIT $2`, vectorLiteral(vector), k)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var out []rag.Match
	for rows.Next() {
		var (
			m     rag.Match
			docID int64
		)
		if err := rows.Scan(&docID, &m.Source, &m.Index, &m.Text, &m.Score); err != nil {
			return nil, err
		}
		m.DocID = strconv.FormatInt(docID, 10)
		out = append(out, m)
	}
	return out, rows.Err()
}

// vectorLiteral formats v in pgvector's text input format: [1,2,3].
func vectorLiteral(v []float32) string {
	var b strings.Builder
	b.Grow(len(v) * 10)
	b.WriteByte('[')
	for i, f := range v {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.FormatFloat(float64(f), 'f', -1, 32))
	}
	b.WriteByte(']')
	return b.String()
}
