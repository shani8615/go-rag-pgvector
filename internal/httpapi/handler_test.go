package httpapi

import (
	"bytes"
	"encoding/json"
	"io"
	"log/slog"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/shani8615/go-rag-pgvector/internal/rag"
)

func newServer() *httptest.Server {
	svc := rag.NewService(rag.HashEmbedder{Dim: 512}, rag.ExtractiveModel{}, rag.NewMemStore(), rag.DefaultConfig)
	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	return httptest.NewServer(New(svc, log))
}

func postJSON(t *testing.T, url string, body any) (*http.Response, map[string]any) {
	t.Helper()
	b, _ := json.Marshal(body)
	res, err := http.Post(url, "application/json", bytes.NewReader(b))
	if err != nil {
		t.Fatal(err)
	}
	defer res.Body.Close()
	var out map[string]any
	_ = json.NewDecoder(res.Body).Decode(&out)
	return res, out
}

func TestIngestThenAsk(t *testing.T) {
	srv := newServer()
	defer srv.Close()

	res, out := postJSON(t, srv.URL+"/v1/documents", map[string]string{
		"source": "handbook.md",
		"text":   "Employees get 20 days of annual leave per year.\n\nThe office is closed on public holidays.",
	})
	if res.StatusCode != http.StatusCreated || out["chunks"].(float64) < 1 {
		t.Fatalf("ingest: %d %v", res.StatusCode, out)
	}

	res, out = postJSON(t, srv.URL+"/v1/ask", map[string]string{"question": "How many days of annual leave do employees get?"})
	if res.StatusCode != http.StatusOK {
		t.Fatalf("ask: %d %v", res.StatusCode, out)
	}
	if !strings.Contains(out["answer"].(string), "20 days") {
		t.Errorf("answer = %v", out["answer"])
	}
	if srcs := out["sources"].([]any); len(srcs) != 1 {
		t.Errorf("sources = %v", srcs)
	}
}

func TestUploadRejectsUnsupportedFileType(t *testing.T) {
	srv := newServer()
	defer srv.Close()

	var buf bytes.Buffer
	mw := multipart.NewWriter(&buf)
	fw, _ := mw.CreateFormFile("file", "malware.exe")
	_, _ = fw.Write([]byte("MZ..."))
	_ = mw.Close()

	res, err := http.Post(srv.URL+"/v1/documents", mw.FormDataContentType(), &buf)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusBadRequest {
		t.Fatalf("want 400, got %d", res.StatusCode)
	}
}

func TestAskValidation(t *testing.T) {
	srv := newServer()
	defer srv.Close()

	for name, q := range map[string]string{
		"empty":    "  ",
		"too long": strings.Repeat("x", maxQuestionChars+1),
	} {
		if res, _ := postJSON(t, srv.URL+"/v1/ask", map[string]string{"question": q}); res.StatusCode != http.StatusBadRequest {
			t.Errorf("%s: want 400, got %d", name, res.StatusCode)
		}
	}
}

func TestAskWithNoDocumentsSaysIDontKnow(t *testing.T) {
	srv := newServer()
	defer srv.Close()

	_, out := postJSON(t, srv.URL+"/v1/ask", map[string]string{"question": "What is the refund policy?"})
	if out["answer"] != rag.NoAnswer {
		t.Errorf("answer = %v", out["answer"])
	}
}
