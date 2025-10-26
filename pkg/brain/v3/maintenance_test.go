package brain

import (
	"math/rand"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Test that low-usage intra-lobe synapses can be migrated (conservative transfer)
func TestMaintain_Migration_Conservative(t *testing.T) {
	rand.Seed(42)

	b := NewBrain()

	// Tune maintenance params for deterministic tests: run maintenance on
	// every call and make migration/growth more likely so the test finishes
	// quickly.
	b.MaintConfig.MaintInterval = 0
	b.MaintConfig.PMigrateBase = 0.05
	b.MaintConfig.GrowthProb = 0.1
	l := NewLobe("L1", nil)
	a := &Neuron{ID: "A", Bias: 1.0, Activation: ActLinear, GroupID: 1}
	b1 := &Neuron{ID: "B", Bias: 0.0, Activation: ActLinear, GroupID: 2}
	c := &Neuron{ID: "C", Bias: 0.0, Activation: ActLinear, GroupID: 3, RecentActivity: 1.0}

	l.AddNeuron(a)
	l.AddNeuron(b1)
	l.AddNeuron(c)
	b.AddLobe(l)

	syn := l.Connect("A", "B", 0.2, true)
	// make synapse look unused
	syn.LastUsed = 0
	syn.Eligibility = 0.0
	syn.Weight = 0.05

	// advance brain tick so inactivity is large
	b.Tick = 5000

	// Call maintenance repeatedly until the original synapse is removed or timeout
	removed := false
	for i := 0; i < 200; i++ {
		b.MaintainStructuralPlasticity()
		if l.GetSynapse(syn.ID) == nil {
			removed = true
			break
		}
	}

	assert.True(t, removed, "expected original synapse to be removed/migrated by maintenance")

	// ensure there is at least one synapse from A to someone (conservative transfer should create new)
	found := false
	for _, s := range l.Synapses {
		if s.From == "A" && s.To != "B" {
			// weight should be scaled (<= original)
			assert.LessOrEqual(t, s.Weight, syn.Weight*1.0)
			found = true
		}
	}
	assert.True(t, found, "expected to find a new synapse from A to a new target")
}

// Test that correlated neurons can lead to new synapse growth (intra-lobe)
func TestMaintain_Growth_CreatesSynapse(t *testing.T) {
	rand.Seed(123)

	b := NewBrain()

	// make growth more likely for the test and run maintenance every call
	b.MaintConfig.MaintInterval = 0
	b.MaintConfig.GrowthProb = 0.2
	l := NewLobe("L2", nil)
	p := &Neuron{ID: "P", GroupID: 1}
	q := &Neuron{ID: "Q", GroupID: 2}

	l.AddNeuron(p)
	l.AddNeuron(q)
	b.AddLobe(l)

	// bump recent activity to exceed growth threshold when multiplied
	p.RecentActivity = 1.0
	q.RecentActivity = 0.6

	// ensure no existing synapse
	for _, s := range l.Synapses {
		if s.From == "P" && s.To == "Q" {
			t.Fatalf("unexpected pre-existing synapse")
		}
	}

	created := false
	// call maintenance repeatedly until a new synapse appears
	for i := 0; i < 500; i++ {
		b.MaintainStructuralPlasticity()
		if l.GetSynapse("P->Q") != nil {
			created = true
			break
		}
	}

	assert.True(t, created, "expected a new synapse to be created between correlated neurons P and Q")
}
