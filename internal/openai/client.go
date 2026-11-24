// Package openai is a minimal client for OpenAI-compatible embeddings and
// chat-completions APIs (OpenAI, Azure-style gateways, Ollama, vLLM, ...).
// It uses only the standard library.
package openai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to an OpenAI-compatible HTTP API.
type Client struct {
	BaseURL    string // e.g. https://api.openai.com/v1
	APIKey     string
	EmbedModel string // e.g. text-embedding-3-small
	ChatModel  string // e.g. gpt-4o-mini
	HTTP       *http.Client
}

// New returns a Client with a sane request timeout.
func New(baseURL, apiKey, embedModel, chatModel string) *Client {
	return &Client{
		BaseURL:    strings.TrimRight(baseURL, "/"),
		APIKey:     apiKey,
		EmbedModel: embedModel,
		ChatModel:  chatModel,
		HTTP:       &http.Client{Timeout: 60 * time.Second},
	}
}

// APIError is a non-2xx response from the API.
type APIError struct {
	Status  int
	Message string
}

func (e *APIError) Error() string {
	return fmt.Sprintf("openai: HTTP %d: %s", e.Status, e.Message)
}

// maxBatch keeps embedding requests well under provider input limits.
const maxBatch = 96

// Embed implements rag.Embedder.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for start := 0; start < len(texts); start += maxBatch {
		end := min(start+maxBatch, len(texts))
		vecs, err := c.embedBatch(ctx, texts[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

func (c *Client) embedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	req := struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}{c.EmbedModel, texts}

	var resp struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := c.post(ctx, "/embeddings", req, &resp); err != nil {
		return nil, err
	}
	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("openai: got %d embeddings for %d inputs", len(resp.Data), len(texts))
	}
	out := make([][]float32, len(texts))
	for _, d := range resp.Data {
		if d.Index < 0 || d.Index >= len(out) || out[d.Index] != nil {
			return nil, fmt.Errorf("openai: unexpected embedding index %d", d.Index)
		}
		out[d.Index] = d.Embedding
	}
	return out, nil
}

// Complete implements rag.ChatModel.
func (c *Client) Complete(ctx context.Context, system, user string) (string, error) {
	type msg struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	req := struct {
		Model       string  `json:"model"`
		Messages    []msg   `json:"messages"`
		Temperature float64 `json:"temperature"`
	}{
		Model:       c.ChatModel,
		Messages:    []msg{{"system", system}, {"user", user}},
		Temperature: 0,
	}
	var resp struct {
		Choices []struct {
			Message msg `json:"message"`
		} `json:"choices"`
	}
	if err := c.post(ctx, "/chat/completions", req, &resp); err != nil {
		return "", err
	}
	if len(resp.Choices) == 0 {
		return "", errors.New("openai: response had no choices")
	}
	return resp.Choices[0].Message.Content, nil
}

func (c *Client) post(ctx context.Context, path string, body, out any) error {
	buf, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.BaseURL+path, bytes.NewReader(buf))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if c.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+c.APIKey)
	}

	res, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("openai: %w", err)
	}
	defer res.Body.Close()

	data, err := io.ReadAll(io.LimitReader(res.Body, 32<<20))
	if err != nil {
		return fmt.Errorf("openai: read body: %w", err)
	}
	if res.StatusCode/100 != 2 {
		var e struct {
			Error struct {
				Message string `json:"message"`
			} `json:"error"`
		}
		msg := strings.TrimSpace(string(data))
		if json.Unmarshal(data, &e) == nil && e.Error.Message != "" {
			msg = e.Error.Message
		}
		if len(msg) > 300 {
			msg = msg[:300]
		}
		return &APIError{Status: res.StatusCode, Message: msg}
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("openai: decode response: %w", err)
	}
	return nil
}
