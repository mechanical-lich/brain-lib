package brain

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestUpdateLobeOrder(t *testing.T) {
	b := NewBrain()

	// Create lobes A, B, C, D
	a := NewLobe("A", nil)
	bL := NewLobe("B", nil)
	c := NewLobe("C", nil)
	d := NewLobe("D", nil)

	b.AddLobe(a)
	b.AddLobe(bL)
	b.AddLobe(c)
	b.AddLobe(d)

	// Connect A->B, A->C, B->D, C->D across lobes (using neurons names arbitrary)
	b.Connect("A", "n1", "B", "n2", 1.0, false)
	b.Connect("A", "n1", "C", "n3", 1.0, false)
	b.Connect("B", "n2", "D", "n4", 1.0, false)
	b.Connect("C", "n3", "D", "n4", 1.0, false)

	order := b.updateOrder
	assert.NotEmpty(t, order)

	pos := func(name string) int {
		for i, v := range order {
			if v == name {
				return i
			}
		}
		return -1
	}

	assert.True(t, pos("A") >= 0)
	assert.True(t, pos("B") >= 0)
	assert.True(t, pos("C") >= 0)
	assert.True(t, pos("D") >= 0)

	assert.True(t, pos("A") < pos("B"), "A should come before B")
	assert.True(t, pos("A") < pos("C"), "A should come before C")
	assert.True(t, pos("B") < pos("D"), "B should come before D")
	assert.True(t, pos("C") < pos("D"), "C should come before D")
}
