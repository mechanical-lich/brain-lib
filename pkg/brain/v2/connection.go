package brain

type Connection struct {
	Name     string  `json:"name"`      // Optional name for the connection
	FromLobe string  `json:"from_lobe"` // Use BRAIN_INPUT for connections from brain inputs
	ToLobe   string  `json:"to_lobe"`   // Use BRAIN_OUTPUT for connections to brain outputs
	FromID   int     `json:"from_id"`   // Index of the output from the FromLobe
	ToID     int     `json:"to_id"`     // Index of the input to the ToLobe
	Weight   float64 `json:"weight"`
}
