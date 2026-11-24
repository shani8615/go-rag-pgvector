// Command server runs the RAG HTTP API.
package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/shani8615/go-rag-pgvector/internal/httpapi"
	"github.com/shani8615/go-rag-pgvector/internal/openai"
	"github.com/shani8615/go-rag-pgvector/internal/pgstore"
	"github.com/shani8615/go-rag-pgvector/internal/rag"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(log); err != nil {
		log.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	var (
		emb  rag.Embedder
		chat rag.ChatModel
	)
	if env("OFFLINE_DEMO", "") == "1" {
		// No API key needed; keyword-based, for trying the API locally only.
		log.Warn("OFFLINE_DEMO=1: using keyword embeddings and extractive answers, not an LLM")
		emb, chat = rag.HashEmbedder{Dim: envInt("EMBED_DIM", 1536)}, rag.ExtractiveModel{}
	} else {
		key := os.Getenv("OPENAI_API_KEY")
		if key == "" {
			return errors.New("OPENAI_API_KEY is not set (or set OFFLINE_DEMO=1)")
		}
		c := openai.New(
			env("OPENAI_BASE_URL", "https://api.openai.com/v1"), key,
			env("EMBED_MODEL", "text-embedding-3-small"),
			env("CHAT_MODEL", "gpt-4o-mini"),
		)
		emb, chat = c, c
	}

	var store rag.Store
	if dsn := os.Getenv("DATABASE_URL"); dsn != "" {
		pg, err := pgstore.Open(ctx, dsn, envInt("EMBED_DIM", 1536))
		if err != nil {
			return fmt.Errorf("postgres: %w", err)
		}
		defer pg.Close()
		store = pg
		log.Info("using postgres + pgvector")
	} else {
		store = rag.NewMemStore()
		log.Warn("DATABASE_URL not set: using in-memory store (data is lost on restart)")
	}

	cfg := rag.DefaultConfig
	if _, ok := chat.(rag.ExtractiveModel); ok {
		// Keyword vectors score lower than real embeddings.
		cfg.MinScore = 0.1
	}
	cfg.TopK = envInt("TOP_K", cfg.TopK)
	if v, err := strconv.ParseFloat(env("MIN_SCORE", ""), 64); err == nil {
		cfg.MinScore = v
	}
	svc := rag.NewService(emb, chat, store, cfg)

	srv := &http.Server{
		Addr:              ":" + env("PORT", "8080"),
		Handler:           httpapi.New(svc, log),
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      120 * time.Second,
	}
	errc := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", srv.Addr)
		errc <- srv.ListenAndServe()
	}()

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if n, err := strconv.Atoi(os.Getenv(k)); err == nil && n > 0 {
		return n
	}
	return def
}
