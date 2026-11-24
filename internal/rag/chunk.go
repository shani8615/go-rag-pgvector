package rag

import (
	"strings"
	"unicode"
	"unicode/utf8"
)

// ChunkOptions controls how documents are split before embedding.
type ChunkOptions struct {
	// MaxChars is the soft upper bound for a chunk, in characters.
	MaxChars int
	// Overlap is how many trailing characters of the previous chunk are
	// repeated at the start of the next one, so answers that straddle a
	// boundary are still retrievable.
	Overlap int
}

// DefaultChunkOptions works well for prose and Markdown with
// OpenAI-style embedding models.
var DefaultChunkOptions = ChunkOptions{MaxChars: 1200, Overlap: 200}

// Chunk splits text into overlapping chunks. It prefers to break on
// paragraph boundaries, then sentence boundaries, and only hard-splits
// words as a last resort. Empty or whitespace-only input yields no chunks.
func Chunk(text string, opt ChunkOptions) []string {
	if opt.MaxChars <= 0 {
		opt = DefaultChunkOptions
	}
	if opt.Overlap < 0 || opt.Overlap >= opt.MaxChars {
		opt.Overlap = 0
	}

	text = normalize(text)
	if text == "" {
		return nil
	}

	var units []string
	for _, p := range strings.Split(text, "\n\n") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if len(p) <= opt.MaxChars {
			units = append(units, p)
			continue
		}
		units = append(units, splitLong(p, opt.MaxChars)...)
	}

	var (
		chunks []string
		cur    strings.Builder
	)
	flush := func() {
		s := strings.TrimSpace(cur.String())
		if s != "" {
			chunks = append(chunks, s)
		}
		cur.Reset()
		if opt.Overlap > 0 && s != "" {
			cur.WriteString(tail(s, opt.Overlap))
		}
	}

	for _, u := range units {
		if cur.Len() > 0 && cur.Len()+2+len(u) > opt.MaxChars {
			flush()
		}
		if cur.Len() > 0 {
			cur.WriteString("\n\n")
		}
		cur.WriteString(u)
	}
	// The final buffer may hold only overlap from the previous chunk; skip it then.
	if s := strings.TrimSpace(cur.String()); s != "" {
		if len(chunks) == 0 || !strings.HasSuffix(chunks[len(chunks)-1], s) {
			chunks = append(chunks, s)
		}
	}
	return chunks
}

func normalize(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	// Collapse 3+ newlines into a paragraph break.
	for strings.Contains(s, "\n\n\n") {
		s = strings.ReplaceAll(s, "\n\n\n", "\n\n")
	}
	return strings.TrimSpace(s)
}

// splitLong breaks an oversized paragraph on sentence ends, then on spaces.
func splitLong(p string, max int) []string {
	var out []string
	for len(p) > max {
		cut := lastBreak(p[:max], ".!?")
		if cut <= max/2 {
			cut = lastBreak(p[:max], " \n\t")
		}
		if cut <= 0 {
			cut = runeBoundary(p, max)
		}
		out = append(out, strings.TrimSpace(p[:cut]))
		p = strings.TrimSpace(p[cut:])
	}
	if p != "" {
		out = append(out, p)
	}
	return out
}

// lastBreak returns the index just after the last rune in s that is one of chars.
func lastBreak(s, chars string) int {
	return strings.LastIndexAny(s, chars) + 1
}

// runeBoundary moves i back to the start of a UTF-8 rune so we never split
// a multi-byte character.
func runeBoundary(s string, i int) int {
	for i > 0 && i < len(s) && !utf8.RuneStart(s[i]) {
		i--
	}
	return i
}

// tail returns roughly the last n characters of s, starting at a word boundary.
func tail(s string, n int) string {
	if len(s) <= n {
		return s
	}
	t := s[runeBoundary(s, len(s)-n):]
	if i := strings.IndexFunc(t, unicode.IsSpace); i >= 0 {
		t = t[i+1:]
	}
	return strings.TrimSpace(t)
}
