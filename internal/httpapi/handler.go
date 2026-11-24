// Package httpapi exposes the RAG service over a small JSON HTTP API.
package httpapi

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"mime"
	"net/http"
	"path/filepath"
	"strings"
	"time"

	"github.com/shani8615/go-rag-pgvector/internal/rag"
)

// Limits keep a single request from exhausting memory or the API budget.
const (
	maxDocBytes      = 2 << 20 // 2 MiB per document
	maxQuestionChars = 1000
	requestTimeout   = 90 * time.Second
)

// RAG is the subset of rag.Service the API needs.
type RAG interface {
	Ingest(ctx context.Context, source, text string) (string, int, error)
	Ask(ctx context.Context, question string) (rag.Answer, error)
}

// New returns the API router.
func New(svc RAG, log *slog.Logger) http.Handler {
	h := &handler{svc: svc, log: log}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("POST /v1/documents", h.ingest)
	mux.HandleFunc("POST /v1/ask", h.ask)
	return mux
}

type handler struct {
	svc RAG
	log *slog.Logger
}

// ingest accepts either JSON {"source": "...", "text": "..."} or a
// multipart upload with a "file" field (.md or .txt).
func (h *handler) ingest(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()
	r.Body = http.MaxBytesReader(w, r.Body, maxDocBytes+64<<10)

	source, text, err := readDocument(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	id, n, err := h.svc.Ingest(ctx, source, text)
	switch {
	case errors.Is(err, rag.ErrEmptyInput):
		writeError(w, http.StatusBadRequest, "document is empty")
		return
	case err != nil:
		h.log.Error("ingest failed", "source", source, "err", err)
		writeError(w, http.StatusBadGateway, "could not index document")
		return
	}
	h.log.Info("document ingested", "doc_id", id, "source", source, "chunks", n)
	writeJSON(w, http.StatusCreated, map[string]any{"doc_id": id, "source": source, "chunks": n})
}

func readDocument(r *http.Request) (source, text string, err error) {
	ct, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	switch ct {
	case "application/json":
		var req struct {
			Source string `json:"source"`
			Text   string `json:"text"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			return "", "", errors.New("invalid JSON body")
		}
		if req.Source == "" {
			req.Source = "inline"
		}
		return req.Source, req.Text, nil

	case "multipart/form-data":
		f, hdr, err := r.FormFile("file")
		if err != nil {
			return "", "", errors.New(`expected a "file" field`)
		}
		defer f.Close()
		ext := strings.ToLower(filepath.Ext(hdr.Filename))
		if ext != ".md" && ext != ".txt" && ext != ".markdown" {
			return "", "", errors.New("only .md and .txt files are supported")
		}
		data, err := io.ReadAll(io.LimitReader(f, maxDocBytes+1))
		if err != nil {
			return "", "", errors.New("could not read file")
		}
		if len(data) > maxDocBytes {
			return "", "", errors.New("file is larger than 2 MiB")
		}
		return filepath.Base(hdr.Filename), string(data), nil
	}
	return "", "", errors.New("use application/json or multipart/form-data")
}

func (h *handler) ask(w http.ResponseWriter, r *http.Request) {
	ctx, cancel := context.WithTimeout(r.Context(), requestTimeout)
	defer cancel()
	r.Body = http.MaxBytesReader(w, r.Body, 16<<10)

	var req struct {
		Question string `json:"question"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid JSON body")
		return
	}
	if len([]rune(req.Question)) > maxQuestionChars {
		writeError(w, http.StatusBadRequest, "question is too long")
		return
	}

	ans, err := h.svc.Ask(ctx, req.Question)
	switch {
	case errors.Is(err, rag.ErrEmptyInput):
		writeError(w, http.StatusBadRequest, "question is empty")
		return
	case err != nil:
		h.log.Error("ask failed", "err", err)
		writeError(w, http.StatusBadGateway, "could not answer right now")
		return
	}
	writeJSON(w, http.StatusOK, ans)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
