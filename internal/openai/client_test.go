package openai

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestEmbedReordersByIndexAndSendsAuth(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/embeddings" {
			t.Errorf("path = %s", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Errorf("auth header = %q", got)
		}
		// Return results out of order; the client must fix that.
		_, _ = w.Write([]byte(`{"data":[{"index":1,"embedding":[0,1]},{"index":0,"embedding":[1,0]}]}`))
	}))
	defer srv.Close()

	c := New(srv.URL, "test-key", "emb", "chat")
	vecs, err := c.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if vecs[0][0] != 1 || vecs[1][1] != 1 {
		t.Fatalf("vectors not reordered by index: %v", vecs)
	}
}

func TestCompleteSendsSystemAndUser(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model    string `json:"model"`
			Messages []struct {
				Role, Content string
			} `json:"messages"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "chat" || len(req.Messages) != 2 || req.Messages[0].Role != "system" {
			t.Errorf("unexpected request: %+v", req)
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"hi [1]"}}]}`))
	}))
	defer srv.Close()

	got, err := New(srv.URL, "", "emb", "chat").Complete(context.Background(), "sys", "user")
	if err != nil || got != "hi [1]" {
		t.Fatalf("got %q, %v", got, err)
	}
}

func TestAPIErrorSurfacesMessage(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error":{"message":"Incorrect API key provided"}}`))
	}))
	defer srv.Close()

	_, err := New(srv.URL, "bad", "emb", "chat").Embed(context.Background(), []string{"x"})
	var apiErr *APIError
	if !errors.As(err, &apiErr) || apiErr.Status != 401 || apiErr.Message != "Incorrect API key provided" {
		t.Fatalf("got %v", err)
	}
}
