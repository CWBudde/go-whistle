package hub

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
)

// DefaultEndpoint is the Hugging Face Hub, used when HF_ENDPOINT is unset.
const DefaultEndpoint = "https://huggingface.co"

// Client talks to the Hugging Face Hub. The zero value works: it uses http.DefaultClient and DefaultEndpoint.
type Client struct {
	HTTP     *http.Client
	Endpoint string    // base URL without trailing slash
	Token    string    // sent as a Bearer token when set
	Offline  bool      // no network: only verify files already on disk
	Progress io.Writer // receives one line per file event; nil discards
}

// NewClientFromEnv configures a Client from HF_TOKEN, HF_HUB_OFFLINE and HF_ENDPOINT, the variables
// huggingface_hub reads.
func NewClientFromEnv() *Client {
	endpoint := strings.TrimRight(os.Getenv("HF_ENDPOINT"), "/")
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}

	return &Client{
		HTTP:     http.DefaultClient,
		Endpoint: endpoint,
		Token:    os.Getenv("HF_TOKEN"),
		Offline:  envTrue(os.Getenv("HF_HUB_OFFLINE")),
	}
}

func envTrue(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// AccessDeniedError reports a 401/403 from the Hub: the repository is gated or private and needs a token.
type AccessDeniedError struct {
	Repo string
}

func (e *AccessDeniedError) Error() string {
	return "access denied for " + e.Repo + "; set HF_TOKEN to a token with access"
}

// ResolveRevision returns the full commit sha that rev (a branch, tag or possibly short commit) points to.
func (c *Client) ResolveRevision(ctx context.Context, repo, rev string) (string, error) {
	u, err := url.JoinPath(c.endpoint(), "api", "models", repo, "revision", rev)
	if err != nil {
		return "", fmt.Errorf("build revision URL: %w", err)
	}

	resp, err := c.get(ctx, u, "")
	if err != nil {
		return "", err
	}

	defer func() { _ = resp.Body.Close() }()

	err = checkStatus(resp, repo, http.StatusOK)
	if err != nil {
		return "", fmt.Errorf("resolve revision %s of %s: %w", rev, repo, err)
	}

	var info struct {
		SHA string `json:"sha"`
	}

	err = json.NewDecoder(resp.Body).Decode(&info)
	if err != nil {
		return "", fmt.Errorf("resolve revision %s of %s: decode: %w", rev, repo, err)
	}

	if !commitPattern.MatchString(info.SHA) {
		return "", fmt.Errorf("resolve revision %s of %s: unexpected sha %q", rev, repo, info.SHA)
	}

	return info.SHA, nil
}

func (c *Client) endpoint() string {
	if c.Endpoint == "" {
		return DefaultEndpoint
	}

	return c.Endpoint
}

func (c *Client) httpClient() *http.Client {
	if c.HTTP == nil {
		return http.DefaultClient
	}

	return c.HTTP
}

func (c *Client) logf(format string, args ...any) {
	if c.Progress != nil {
		_, _ = fmt.Fprintf(c.Progress, format+"\n", args...)
	}
}

// get sends a GET with the token and an optional Range header. Go's http.Client drops the Authorization header
// when the Hub redirects to its CDN on another host.
func (c *Client) get(ctx context.Context, u, rangeHdr string) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}

	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}

	if rangeHdr != "" {
		req.Header.Set("Range", rangeHdr)
	}

	// #nosec G704 -- the URL is built from the configured Hub endpoint and pinned manifest data.
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("GET %s: %w", u, err)
	}

	return resp, nil
}

// checkStatus maps 401/403 to *AccessDeniedError and any status not in ok to an error.
func checkStatus(resp *http.Response, repo string, ok ...int) error {
	if slices.Contains(ok, resp.StatusCode) {
		return nil
	}

	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		return &AccessDeniedError{Repo: repo}
	}

	return fmt.Errorf("unexpected status %s", resp.Status)
}
