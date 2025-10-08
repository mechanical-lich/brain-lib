package brain

import (
	"encoding/json"
	"os"
)

type Brain struct {
	Lobes       []*Lobe                 `json:"lobes"`
	lobesLookup map[string]*Lobe        `json:"-"`
	Connections map[string][]Connection `json:"connections"` // Keyed by lobe name
}

func NewBrain() *Brain {
	return &Brain{
		Lobes:       make([]*Lobe, 0),
		lobesLookup: make(map[string]*Lobe),
		Connections: make(map[string][]Connection),
	}
}

func (b *Brain) NewLobe(name string, numInputs int, numOutputs int) *Lobe {
	lobe := NewLobe(name, numInputs, numOutputs, nil)
	b.Lobes = append(b.Lobes, lobe)
	b.lobesLookup[name] = lobe
	b.Connections[name] = make([]Connection, 0)
	return lobe
}

func (b *Brain) AddLobe(lobe *Lobe) {
	b.Lobes = append(b.Lobes, lobe)
	b.lobesLookup[lobe.Name] = lobe
	b.Connections[lobe.Name] = make([]Connection, 0)
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
	b.Connections[toLobe] = append(b.Connections[toLobe], conn)
}

func (b *Brain) RemoveConnection(fromLobe string, toLobe string, fromID int, toID int) {
	connections := b.Connections[toLobe]
	newConnections := make([]Connection, 0, len(connections))
	for _, conn := range connections {
		if !(conn.FromLobe == fromLobe && conn.ToLobe == toLobe && conn.FromID == fromID && conn.ToID == toID) {
			newConnections = append(newConnections, conn)
		}
	}
	b.Connections[toLobe] = newConnections
}

func (b *Brain) WeightedConnect(name string, fromLobe string, toLobe string, fromID int, toID int, weight float64) {
	conn := Connection{Name: name, FromLobe: fromLobe, ToLobe: toLobe, FromID: fromID, ToID: toID, Weight: weight}
	b.Connections[toLobe] = append(b.Connections[toLobe], conn)
}

func (b *Brain) Think() {
	// Loop through lobes, set any missing inputs from connections, then process
	for _, lobe := range b.Lobes {
		for _, conn := range b.Connections[lobe.Name] {
			if fromLobe, exists := b.lobesLookup[conn.FromLobe]; exists {
				fromOutputs := fromLobe.Outputs
				if conn.FromID < len(fromOutputs) && conn.ToID < len(lobe.Inputs) {
					lobe.Inputs[conn.ToID] += fromOutputs[conn.FromID] * conn.Weight
				}
			}
		}

		lobe.Process()
	}
}

func (b *Brain) refreshLookup() {
	b.lobesLookup = make(map[string]*Lobe)
	for _, lobe := range b.Lobes {
		b.lobesLookup[lobe.Name] = lobe
	}
}

func (b *Brain) Save(filename string) error {
	// Marsha l the Brain struct to JSON
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}

	// Write the JSON data to the specified file
	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return err
	}

	return nil
}
func LoadBrain(filename string) (*Brain, error) {
	// Read the JSON data from the specified file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}

	// Unmarshal the JSON data into a Brain struct
	var brain Brain
	err = json.Unmarshal(data, &brain)
	if err != nil {
		return nil, err
	}

	brain.refreshLookup()

	return &brain, nil
}

func LoadBrainFromString(data string) (*Brain, error) {
	// Unmarshal the JSON data into a Brain struct
	var brain Brain
	err := json.Unmarshal([]byte(data), &brain)
	if err != nil {
		return nil, err
	}

	brain.refreshLookup()

	return &brain, nil
}

func (b *Brain) ToJSON() (string, error) {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}
