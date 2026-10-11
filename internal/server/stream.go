package server

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/url"
	"time"

	"github.com/coder/websocket"
	"github.com/cwbudde/go-whistle/internal/pcm"
	"github.com/cwbudde/go-whistle/whistle"
)

// writeTimeout bounds the write of one result to a /stream client.
const writeTimeout = 10 * time.Second

// stream is GET /stream: one whistle.Stream per WebSocket connection.
func (s *Server) stream(w http.ResponseWriter, r *http.Request) {
	opts, width, err := streamQuery(r.URL.Query())
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())

		return
	}

	ok, why := s.open()
	if !ok {
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusServiceUnavailable, why)

		return
	}

	defer s.close()

	st, err := s.m.NewStream(opts)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())

		return
	}

	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{OriginPatterns: s.cfg.OriginPatterns})
	if err != nil {
		return // Accept has answered
	}

	defer func() { _ = c.CloseNow() }()

	// One message is one Process call, at most 30 s; one more whole sample
	// closes the connection with 1009. The limit admits a partial sample
	// after the 30 s, so that it gets the 1007 of any partial sample (PR #35
	// review).
	c.SetReadLimit(int64(whistle.MaxSamples*width + width - 1))

	// The session's context ends with the session, or when Shutdown's
	// deadline drops it; the request's own ends with the upgrade.
	ctx, cancel := context.WithCancel(context.WithoutCancel(r.Context()))
	defer cancel()

	go func() {
		select {
		case <-s.dropping:
			cancel()

			_ = c.CloseNow()
		case <-ctx.Done():
		}
	}()

	ss := &session{s: s, c: c, st: st, width: width, cancel: cancel}
	ss.run(ctx)
}

// streamQuery reads the options and the PCM format of /stream.
func streamQuery(q url.Values) (whistle.StreamOptions, int, error) {
	err := checkQuery(q, "format", "language", "keywords")
	if err != nil {
		return whistle.StreamOptions{}, 0, err
	}

	format := q.Get("format")
	if format == "" {
		format = pcm.S16LE
	}

	width, err := pcm.Width(format)
	if err != nil {
		return whistle.StreamOptions{}, 0, err
	}

	return whistle.StreamOptions{Language: q.Get("language"), Keywords: q["keywords"]}, width, nil
}

// session is one /stream connection.
type session struct {
	s     *Server
	c     *websocket.Conn
	st    Stream
	width int
	// cancel ends the session's context.
	cancel context.CancelFunc
}

// message is one message the client sent, or the error that ended reading.
type message struct {
	typ  websocket.MessageType
	data []byte
	err  error
}

// end is how a session ends: with a close code and reason, after the Stop
// result when stop is set. The zero end drops the connection.
type end struct {
	code   websocket.StatusCode
	reason string
	stop   bool
}

// run reads the messages apart, so that a shutdown or the idle timer can end
// a session whose client is silent, and handles them until the session ends.
func (ss *session) run(ctx context.Context) {
	msgs := make(chan message)
	done := make(chan struct{})

	go func() {
		for {
			typ, data, err := ss.c.Read(ctx)
			if err != nil {
				// The client left or the connection failed. The handler may
				// wait for a pass and not take the message for long, so the
				// context ends here: the wait gives up and the session ends
				// (PR #35 review).
				ss.cancel()
			}

			select {
			case msgs <- message{typ, data, err}:
			case <-done:
				return
			}

			if err != nil {
				return
			}
		}
	}()

	e := ss.loop(ctx, msgs)

	close(done)

	if e.code == 0 {
		return
	}

	if e.stop && ss.send(ctx, ss.st.Stop()) != nil {
		return
	}

	_ = ss.c.Close(e.code, e.reason)
}

// loop handles messages until the client stops or leaves, the session
// idles or the server shuts down.
func (ss *session) loop(ctx context.Context, msgs <-chan message) end {
	idle := time.NewTimer(ss.s.cfg.StreamIdle)
	defer idle.Stop()

	for {
		select {
		case <-ss.s.stopping:
			return end{websocket.StatusGoingAway, "the server is shutting down", true}
		case <-idle.C:
			return end{websocket.StatusPolicyViolation, "idle for " + ss.s.cfg.StreamIdle.String(), true}
		case m := <-msgs:
			if m.err != nil {
				return end{} // the client closed or the connection failed: nobody to send Stop to
			}

			e, more := ss.handle(ctx, m)
			if !more {
				return e
			}

			idle.Reset(ss.s.cfg.StreamIdle)
		}
	}
}

// handle runs one message: Process on a PCM chunk and its result, or the
// stop. false ends the session with the end returned.
func (ss *session) handle(ctx context.Context, m message) (end, bool) {
	if m.typ == websocket.MessageText {
		if isStop(m.data) {
			return end{websocket.StatusNormalClosure, "", true}, false
		}

		return end{code: websocket.StatusUnsupportedData, reason: `want binary PCM or {"type":"stop"}`}, false
	}

	x, err := pcm.Decode(m.data, ss.width)
	if err != nil {
		return end{code: websocket.StatusInvalidFramePayloadData, reason: "the PCM chunk ends inside a sample"}, false
	}

	err = ss.s.gate.enter(ctx, false)
	if err != nil {
		return end{}, false // the client left, or dropped at the shutdown deadline
	}

	res, err := func() (whistle.StreamResult, error) {
		defer ss.s.gate.leave()

		return ss.st.Process(x)
	}()
	if err != nil {
		if ss.s.cfg.Log != nil {
			ss.s.cfg.Log.Printf("stream: process: %v", err)
		}

		return end{code: websocket.StatusInternalError, reason: "process failed"}, false
	}

	if ss.send(ctx, res) != nil {
		return end{}, false
	}

	return end{}, true
}

// errDropped reports a session dropped at Shutdown's deadline.
var errDropped = errors.New("dropped at the shutdown deadline")

// send writes r's stream JSON as a text message. A dropped session sends
// nothing, even when its pass ends before the drop has closed the connection.
func (ss *session) send(ctx context.Context, r whistle.StreamResult) error {
	select {
	case <-ss.s.dropping:
		return errDropped
	default:
	}

	line, err := r.MarshalJSON()
	if err != nil {
		return err
	}

	ctx, cancel := context.WithTimeout(ctx, writeTimeout)
	defer cancel()

	return ss.c.Write(ctx, websocket.MessageText, line)
}

// isStop reports whether a text message is {"type":"stop"}.
func isStop(data []byte) bool {
	var v struct {
		Type string `json:"type"`
	}

	return json.Unmarshal(data, &v) == nil && v.Type == "stop"
}
