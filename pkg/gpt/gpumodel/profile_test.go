package gpumodel

import (
	"fmt"
	"math/rand"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/model"
	"github.com/mechanical-lich/brain-lib/pkg/gpt/train"
)

// TestProfileStep prints where a training step's time goes at the BPE
// shape. Run it with -v to see the table; it is skipped in -short mode.
func TestProfileStep(t *testing.T) {
	if testing.Short() {
		t.Skip("profile only")
	}
	d := dev(t)
	cfg := model.Config{VocabSize: 2000, ContextLen: 256, NumLayers: 6, NumHeads: 6, EmbedDim: 384, Dropout: 0.1}
	g := model.New(cfg, rand.New(rand.NewSource(1)))
	m, err := New(d, cfg)
	require.NoError(t, err)
	defer m.Close()
	require.NoError(t, m.LoadFrom(g))
	m.SetTraining(true)
	rng := rand.New(rand.NewSource(2))
	x := randomTokens(rng, 16, 256, cfg.VocabSize)
	y := randomTokens(rng, 16, 256, cfg.VocabSize)
	oc := train.DefaultAdamW()
	// warm up pipelines and bind groups, then time a plain step and a profiled one
	_, err = m.TrainStep(x, y, oc, 1, 1)
	require.NoError(t, err)
	start := time.Now()
	const reps = 3
	for i := 0; i < reps; i++ {
		_, err = m.TrainStep(x, y, oc, 1, i+2)
		require.NoError(t, err)
	}
	plain := time.Since(start) / reps
	phases, err := m.Profile(x, y, oc, 1, reps+2)
	require.NoError(t, err)
	var total time.Duration
	for _, ph := range phases {
		total += ph.Time
	}
	sort.Slice(phases, func(i, j int) bool { return phases[i].Time > phases[j].Time })
	var sb string
	sb += fmt.Sprintf("plain TrainStep %v; profiled phases sum %v (%s)\n", plain.Round(time.Millisecond), total.Round(time.Millisecond), d.Name())
	for _, ph := range phases {
		sb += fmt.Sprintf("  %-22s %7.1f ms  %5.1f%%\n", ph.Name, float64(ph.Time.Microseconds())/1000, 100*float64(ph.Time)/float64(total))
	}
	t.Log("\n" + sb)
}
