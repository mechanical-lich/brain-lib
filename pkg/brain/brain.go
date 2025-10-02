package brain

import (
	"log/slog"
)

// A Connection represents a directed link between two lobes in the brain or between brain inputs and the lobes.
const BRAIN_INPUT = "BRAIN_INPUT"
const BRAIN_OUTPUT = "BRAIN_OUTPUT"

type Connection struct {
	Name     string // Optional name for the connection
	FromLobe string // Use BRAIN_INPUT for connections from brain inputs
	ToLobe   string // Use BRAIN_OUTPUT for connections to brain outputs
	FromID   int    // Index of the output from the FromLobe
	ToID     int    // Index of the input to the ToLobe
	Weight   float64
}

type Brain struct {
	lobes       []*Lobe
	lobesLookup map[string]*Lobe
	connections map[string][]Connection // Keyed by lobe name
	numInputs   int
	numOutputs  int
}

func NewBrain(numInputs int, numOutputs int) *Brain {
	return &Brain{
		lobes:       make([]*Lobe, 0),
		lobesLookup: make(map[string]*Lobe),
		connections: make(map[string][]Connection),
		numInputs:   numInputs,
		numOutputs:  numOutputs,
	}
}

func (b *Brain) NewLobe(name string, numInputs int, numOutputs int) *Lobe {
	lobe := NewLobe(name, numInputs, numOutputs, nil)
	b.lobes = append(b.lobes, lobe)
	b.lobesLookup[name] = lobe
	b.connections[name] = make([]Connection, 0)
	return lobe
}

func (b *Brain) AddLobe(lobe *Lobe) {
	b.lobes = append(b.lobes, lobe)
	b.lobesLookup[lobe.Name] = lobe
	b.connections[lobe.Name] = make([]Connection, 0)
}

func (b *Brain) GetLobe(name string) *Lobe {
	lobe, exists := b.lobesLookup[name]

	if !exists {
		return nil
	}

	return lobe
}

func (b *Brain) Connect(name string, fromLobe string, toLobe string, fromID int, toID int) {
	conn := Connection{Name: name, FromLobe: fromLobe, ToLobe: toLobe, FromID: fromID, ToID: toID, Weight: 1.0}
	b.connections[toLobe] = append(b.connections[toLobe], conn)
}

func (b *Brain) WeightedConnect(name string, fromLobe string, toLobe string, fromID int, toID int, weight float64) {
	conn := Connection{Name: name, FromLobe: fromLobe, ToLobe: toLobe, FromID: fromID, ToID: toID, Weight: weight}
	b.connections[toLobe] = append(b.connections[toLobe], conn)
}

func (b *Brain) Think(inputs []float64) []float64 {
	var outputs = make([]float64, b.numOutputs)

	for _, lobe := range b.lobes {
		lobeInputs := make([]float64, lobe.numInputs)
		for _, conn := range b.connections[lobe.Name] {
			if conn.FromLobe == BRAIN_INPUT {
				if conn.FromID < len(inputs) && conn.ToID < len(lobeInputs) {
					lobeInputs[conn.ToID] = inputs[conn.FromID]
				}
			} else if fromLobe, exists := b.lobesLookup[conn.FromLobe]; exists {
				fromOutputs := fromLobe.GetOutputs()
				if conn.FromID < len(fromOutputs) && conn.ToID < len(lobeInputs) {
					lobeInputs[conn.ToID] += fromOutputs[conn.FromID] * conn.Weight
				}
			}
		}

		lobe.Process(lobeInputs)
	}

	// Collect outputs from lobes connected to brain outputs
	for _, conn := range b.connections[BRAIN_OUTPUT] {
		if fromLobe, exists := b.lobesLookup[conn.FromLobe]; exists {
			fromOutputs := fromLobe.GetOutputs()

			if conn.ToID < len(outputs) && conn.FromID < len(fromOutputs) {
				outputs[conn.ToID] += fromOutputs[conn.FromID] * conn.Weight
			} else {
				slog.Warn("Connection indices out of range", "conn", conn, "len(outputs)", len(outputs), "len(fromOutputs)", len(fromOutputs))
			}
		}
	}

	return outputs
}
