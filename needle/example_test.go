package needle_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"

	"github.com/cwbudde/go-whistle/needle"
)

// The directory `just fetch-needle3` fills. The examples that load the
// model compile under go test but do not run.
const modelDir = "models/needle3"

// tools is a tool set as NewAgent takes it.
var tools = []byte(`[{"name":"set_timer","description":"Start a countdown timer",` +
	`"parameters":{"type":"object","properties":{"minutes":{"type":"integer"}},"required":["minutes"]}}]`)

// Load the model once and answer a request with the engine's envelope.
func Example() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := needle.Load(modelDir)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	a, err := m.NewAgent(tools, needle.AgentOptions{})
	if err != nil {
		log.Print(err)

		return
	}

	r, err := a.Complete("set a timer for 10 minutes", needle.CompleteOptions{})
	if err != nil {
		log.Print(err)

		return
	}

	for _, c := range r.FunctionCalls {
		fmt.Println(c.Name, string(c.Arguments))
	}

	js, err := json.Marshal(r)
	if err != nil {
		log.Print(err)

		return
	}

	fmt.Println(string(js))
}

// Run the calls with Go functions until the model answers without one.
func ExampleAgent_Run() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := needle.Load(modelDir)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	a, err := m.NewAgent(tools, needle.AgentOptions{AutoDate: true})
	if err != nil {
		log.Print(err)

		return
	}

	r, err := a.Run("set a timer for 10 minutes", map[string]needle.ToolFunc{
		"set_timer": func(args json.RawMessage) (any, error) {
			var in struct {
				Minutes int `json:"minutes"`
			}

			err := json.Unmarshal(args, &in)
			if err != nil {
				return nil, err
			}

			return map[string]any{"started": true, "minutes": in.Minutes}, nil
		},
	}, needle.RunOptions{})
	if err != nil {
		log.Print(err)

		return
	}

	fmt.Println(r.Type, r.Results)
}

// Fill a schema from a text, checked against the text.
func ExampleModel_ExtractInto() { //nolint:testableexamples // needs the model, which go test may not have
	m, err := needle.Load(modelDir)
	if err != nil {
		log.Fatal(err)
	}

	defer func() { _ = m.Close() }()

	schema := []byte(`{"name":"contact","parameters":{"type":"object","properties":{` +
		`"name":{"type":"string"},"city":{"type":"string"}},"required":["name"]}}`)

	var contact struct {
		Name string `json:"name"`
		City string `json:"city"`
	}

	ok, err := m.ExtractInto("Anna Weber moved to Hamburg last year.", schema, &contact, needle.ExtractOptions{})

	var ve *needle.ExtractionValidationError

	switch {
	case errors.As(err, &ve):
		fmt.Println("not grounded:", ve.Fields)
	case err != nil:
		log.Print(err)
	case ok:
		fmt.Println(contact.Name, contact.City)
	}
}
