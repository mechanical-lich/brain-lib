package brain

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// learningBrain builds a two-lobe brain with one plastic inter-lobe synapse and
// a constantly-firing source, so a fixed span of simulated time produces a
// predictable amount of learning.
func learningBrain() (*Brain, *Synapse) {
	b := NewBrain()
	src := NewLobe("src", nil)
	dst := NewLobe("dst", nil)
	src.AddNeuron(&Neuron{ID: "a", Bias: 1.0, Activation: ActLinear})
	dst.AddNeuron(&Neuron{ID: "b", Bias: 0.5, Activation: ActLinear})
	b.AddLobe(src)
	b.AddLobe(dst)

	syn := b.Connect("src", "a", "dst", "b", 0.0, true)
	// Effectively unbounded: with a normal [-2,2] clip the old frame-rate-
	// dependent behaviour saturated at every step size, so every result tied at
	// the clip and the divergence was invisible. The clip is tested elsewhere;
	// here we want the raw magnitude.
	syn.MinWeight, syn.MaxWeight = -1e6, 1e6
	b.Reinforce(Dopamine, 1.0)
	b.Modulators[Dopamine].DecayRate = 0 // hold the signal steady
	return b, syn
}

// runFor simulates `seconds` of time at the given step size.
func runFor(b *Brain, seconds, dt float64) {
	for i := 0; i < int(math.Round(seconds/dt)); i++ {
		b.Step(dt)
	}
}

// The headline regression: the same simulated second must produce the same
// learning whatever the frame rate. Before traces and reinforcement were
// dt-scaled, one second gave a weight change of 0.29 at 15 TPS and a
// clip-pinned 1.0 at 30 TPS and above — so a frame hiccup changed what a
// critter learned.
func TestLearningIsFrameRateIndependent(t *testing.T) {
	var results []float64
	for _, dt := range []float64{1.0 / 15, 1.0 / 30, 1.0 / 60, 1.0 / 120, 1.0 / 240} {
		b, syn := learningBrain()
		runFor(b, 1.0, dt)
		results = append(results, b.Synapses[syn.ID].Weight)
	}

	reference := results[len(results)-1] // finest step size
	require.NotZero(t, reference, "the fixture should actually learn something")

	for i, got := range results {
		rel := math.Abs(got-reference) / math.Abs(reference)
		assert.Less(t, rel, 0.25,
			"result %d (%.6f) differs from the fine-grained result (%.6f) by %.1f%% - learning still depends on frame rate",
			i, got, reference, rel*100)
	}
}

// Traces must converge too, not just the resulting weights.
func TestTraceIsFrameRateIndependent(t *testing.T) {
	var traces []float64
	for _, dt := range []float64{1.0 / 15, 1.0 / 60, 1.0 / 240} {
		b, syn := learningBrain()
		runFor(b, 1.0, dt)
		traces = append(traces, b.Synapses[syn.ID].Trace)
	}

	reference := traces[len(traces)-1]
	require.Greater(t, reference, 0.0)
	for i, got := range traces {
		rel := math.Abs(got-reference) / reference
		assert.Less(t, rel, 0.25, "trace %d (%.4f) vs %.4f", i, got, reference)
	}
}

// The eligibility trace is a leaky integrator, so it settles at roughly
// input*tau rather than growing without bound. Under the old per-frame form it
// reached 326 on the shipped brain, which meant one reward saturated a weight
// to its clip instantly.
func TestTraceSaturatesNearTimeConstant(t *testing.T) {
	b, syn := learningBrain()
	b.Lobes["dst"].PlasticityConfig.TraceDecay = 1.0 // tau = 1s
	// Restore a realistic weight clip. The unbounded fixture above would let
	// the weight, and therefore the target's activation, run away — trace feeds
	// the weight which feeds the activation which feeds the trace.
	syn.MinWeight, syn.MaxWeight = -2, 2

	runFor(b, 10.0, 1.0/60) // long enough to settle

	trace := b.Synapses[syn.ID].Trace
	// Source activation ~1, target ~0.5+, so the product is order 1 and the
	// steady state should be order tau, not order hundreds.
	assert.Less(t, trace, 10.0, "trace should settle near input*tau, not grow unbounded")
	assert.Greater(t, trace, 0.0)
}

// exp(-rate*dt) is exact at any step size, where `x -= x*rate*dt` diverged
// above rate*dt == 2: a modulator with DecayRate 3 at dt 1.0 went
// 1 -> -2 -> 4 -> -8 -> 16, and a runaway dopamine level slammed every plastic
// weight to its clip.
func TestDecayNeverDiverges(t *testing.T) {
	for _, tc := range []struct {
		rate, dt float64
	}{
		{3.0, 1.0}, {10.0, 1.0}, {100.0, 0.5}, {1.0, 60.0},
	} {
		m := &Modulator{Name: Dopamine, Level: 1.0, DecayRate: tc.rate}
		prev := m.Level
		for i := 0; i < 50; i++ {
			m.Step(tc.dt)
			require.LessOrEqual(t, m.Level, prev,
				"rate=%v dt=%v: decay must be monotonically non-increasing", tc.rate, tc.dt)
			require.GreaterOrEqual(t, m.Level, 0.0,
				"rate=%v dt=%v: decay must never go negative", tc.rate, tc.dt)
			prev = m.Level
		}
	}
}

// Decay over a fixed span of time must not depend on how it is subdivided.
func TestDecayIsStepSizeInvariant(t *testing.T) {
	const rate = 2.0
	var results []float64
	for _, dt := range []float64{1.0, 0.5, 0.1, 0.01} {
		m := &Modulator{Name: Dopamine, Level: 1.0, DecayRate: rate}
		runForModulator(m, 1.0, dt)
		results = append(results, m.Level)
	}

	expected := math.Exp(-rate * 1.0)
	for i, got := range results {
		assert.InDelta(t, expected, got, 1e-9,
			"result %d: one second of decay at rate %v should be exp(-2) regardless of step size", i, rate)
	}
}

func runForModulator(m *Modulator, seconds, dt float64) {
	for i := 0; i < int(math.Round(seconds/dt)); i++ {
		m.Step(dt)
	}
}

// A receptor cannot deliver more than the pool holds. UptakeRate 5.0 against a
// 1.0 pool used to deliver 5.0 to the target while removing only 1.0.
func TestReceptorCannotDeliverMoreThanThePoolHolds(t *testing.T) {
	r := &ChemicalReceptor{ChemicalName: "x", UptakeRate: 5.0, ReceptorType: ReceptorTypeExcitatory}
	chem := map[string]float64{"x": 1.0}

	out, consumed := r.GetOutput(chem, 1.0)

	assert.LessOrEqual(t, out, 1.0, "delivered more than the pool contained")
	assert.LessOrEqual(t, consumed, 1.0, "consumed more than the pool contained")
	assert.InDelta(t, out, consumed, 1e-9, "an excitatory receptor should deliver what it consumes")
}

// When a memory ends must not depend on the frame rate beyond the one step the
// snap can lag the analytic crossing: ln(20) ≈ 3.0s at rate 1/s under a 0.05
// threshold, at every step size.
func TestMemoryEndIsFrameRateIndependent(t *testing.T) {
	want := math.Log(20)
	for _, dt := range []float64{1.0 / 15, 1.0 / 60, 1.0 / 240, 0.1} {
		steps := stepsUntilZero(memoryLobe(0.05), dt, int(10/dt))
		require.Positive(t, steps, "dt %.4f: the memory never ended", dt)
		got := float64(steps) * dt
		assert.InDelta(t, want, got, dt+1e-9, "dt %.4f: ended after %.3fs", dt, got)
	}
}
