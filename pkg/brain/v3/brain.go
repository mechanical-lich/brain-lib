package brain

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"os"
)

type Brain struct {
	Lobes       map[string]*Lobe    `json:"lobes"`
	updateOrder []string            `json:"-"`
	Synapses    map[string]*Synapse `json:"synapses"`
	// Tick is a simple step counter used for structural-plasticity bookkeeping
	Tick int `json:"tick"`
	// MaintConfig holds tunable parameters for the structural-plasticity maintenance pass
	MaintConfig MaintenanceConfig `json:"maint_config,omitempty"`
	// lastMaintTick is the last global Tick when maintenance actually ran.
	// It is unexported and not serialized; used to enforce a maintenance interval.
	lastMaintTick int `json:"-"`
}

func NewBrain() *Brain {
	b := &Brain{
		Lobes:    make(map[string]*Lobe),
		Synapses: make(map[string]*Synapse),
	}
	// sensible defaults for maintenance parameters
	b.MaintConfig = MaintenanceConfig{
		MaintenanceLimit:     50,
		WScale:               0.25,
		PMigrateBase:         0.02,
		InactivityThreshold:  20000,
		EligibilityThreshold: 0.005,
		GrowthCorrThreshold:  0.5,
		GrowthProb:           0.02,
		Cooldown:             50,
		// By default, run maintenance every 100 ticks to avoid excessive
		// structural changes when the simulation is stepped every frame.
		MaintInterval: 100,
	}
	return b
}

// MaintenanceConfig contains tunable parameters used by MaintainStructuralPlasticity.
type MaintenanceConfig struct {
	// MaintenanceLimit is the maximum number of rewiring/migration actions
	// the maintenance pass will perform in a single call. Increase to allow
	// more structural changes per tick; decrease to limit network churn.
	MaintenanceLimit int `json:"maintenance_limit"`
	// WScale is the weight scaling factor applied when a synapse is
	// conservatively migrated (transferred to a new target). New synapse
	// weight = old weight * WScale. Typical range: 0.0 - 1.0 (e.g. 0.2-0.5).
	WScale float64 `json:"w_scale"`
	// PMigrateBase is the base probability multiplier for initiating a
	// migration for a candidate synapse. The final migration probability is
	// PMigrateBase * mobility where mobility is a heuristic based on age,
	// inactivity and eligibility. Larger values increase migration frequency.
	PMigrateBase float64 `json:"p_migrate_base"`
	// InactivityThreshold is the number of global ticks after which a
	// synapse is considered inactive. Use larger values to be more patient
	// before moving rarely-used synapses.
	InactivityThreshold int `json:"inactivity_threshold"`
	// EligibilityThreshold is the threshold on the synapse eligibility trace
	// below which the synapse is considered a candidate for migration.
	EligibilityThreshold float64 `json:"eligibility_threshold"`
	// GrowthCorrThreshold is the minimum (pre.RecentActivity *
	// post.RecentActivity) value required to consider growing a new
	// synapse between two neurons. Higher values make growth more selective.
	GrowthCorrThreshold float64 `json:"growth_corr_threshold"`
	// GrowthProb is the probability of creating a synapse when the
	// correlation exceeds GrowthCorrThreshold. Use this to tune how often
	// candidate correlations lead to new connections.
	GrowthProb float64 `json:"growth_prob"`
	// GrowthMinActivity is the minimum recent-activity value a neuron must
	// have to be considered for growth. Both pre and post neurons must
	// exceed this to be eligible. Set to 0 to disable this check.
	GrowthMinActivity float64 `json:"growth_min_activity"`
	// Cooldown sets the number of steps a newly created/migrated synapse
	// will be immune from further maintenance actions. Helps avoid rapid
	// oscillation of rewires. Measured in global ticks.
	Cooldown int `json:"cooldown"`
	// MaintInterval sets the minimum number of global ticks between
	// maintenance runs. If >0, maintenance will only execute when at
	// least MaintInterval ticks have passed since the last maintenance.
	// Set to 0 to run maintenance on every call (useful for tests).
	MaintInterval int `json:"maint_interval"`
}

func (b *Brain) ApplyChemistry(chemicals map[string]float64, dt float64) map[string]float64 {
	for _, lobe := range b.Lobes {
		chemicals = lobe.UpdateChemicalReceptors(dt, chemicals)
	}
	return chemicals
}

func (b *Brain) Step(dt float64) {
	// advance global tick once per simulation step
	b.Tick++
	// Single-pass: for each lobe in topological order, inject external inputs from
	// already-stepped source lobes (via cross-lobe synapses) then step the lobe once.
	for _, lobeId := range b.updateOrder {
		lobe, ok := b.Lobes[lobeId]
		if !ok {
			continue
		}

		// Inject cross-lobe inputs from source lobes that should already have been stepped
		for _, syn := range b.Synapses {
			if syn.ToLobe != lobe.Name {
				continue
			}
			// skip if synapse doesn't reference a source lobe (shouldn't happen)
			if syn.FromLobe == "" {
				continue
			}
			fromLobe, ok1 := b.Lobes[syn.FromLobe]
			if !ok1 {
				continue
			}
			fromNeuron, ok2 := fromLobe.Neurons[syn.From]
			toNeuron, ok3 := lobe.Neurons[syn.To]
			if !ok2 || !ok3 {
				continue
			}
			toNeuron.ExternalInput += fromNeuron.ActivationVal * syn.Weight
		}

		// Now step this lobe once so it consumes external inputs and computes activations
		lobe.Step(dt, b.Tick)
	}
}

// MaintainStructuralPlasticity runs a maintenance pass that may migrate (rewire)
// low-usage synapses (conservative transfer, scaling weight) and grow new synapses
// between neuron pairs with high recent correlation.
func (b *Brain) MaintainStructuralPlasticity() {
	now := b.Tick

	// Check maintenance interval: if MaintInterval > 0 and not enough ticks
	// have passed since the last maintenance run, skip.
	if b.MaintConfig.MaintInterval > 0 {
		if now-b.lastMaintTick < b.MaintConfig.MaintInterval {
			return
		}
	}

	// Tunable parameters (from brain MaintConfig)
	cfg := b.MaintConfig
	maintenanceLimit := cfg.MaintenanceLimit
	wScale := cfg.WScale
	pMigrateBase := cfg.PMigrateBase
	inactivityThreshold := cfg.InactivityThreshold
	eligibilityThreshold := cfg.EligibilityThreshold
	growthCorrThreshold := cfg.GrowthCorrThreshold
	growthProb := cfg.GrowthProb
	growthMinActivity := cfg.GrowthMinActivity

	rewires := 0

	// Intra-lobe maintenance and growth
	for _, l := range b.Lobes {
		// collect synapses slice to avoid map mutation during iteration
		for sid, s := range l.Synapses {
			if rewires >= maintenanceLimit {
				break
			}
			if s.Cooldown > 0 {
				continue
			}
			inactivity := now - s.LastUsed
			mobility := 0.0
			if inactivity > inactivityThreshold {
				mobility += 1.0
			}
			if s.Eligibility < eligibilityThreshold {
				mobility += 1.0
			}
			if math.Abs(s.Weight) < 0.01 {
				mobility += 1.0
			}

			p := pMigrateBase * mobility
			if rand.Float64() < p {
				// choose a new target within same lobe (respect intra-lobe constraint)
				// enforce group constraint via allowedGroupRoute helper
				fromNeuron, okFrom := l.Neurons[s.From]
				if !okFrom {
					continue
				}
				candidates := make([]string, 0)
				for nid := range l.Neurons {
					if nid == s.To {
						continue
					}
					// avoid self-target (don't rewire pre->pre)
					if nid == s.From {
						continue
					}
					// skip if synapse already exists from this pre to candidate
					exists := false
					for _, ss := range l.Synapses {
						if ss.From == s.From && ss.To == nid {
							exists = true
							break
						}
					}
					if exists {
						continue
					}
					// enforce group constraint using allowedGroupRoute
					candNeuron := l.Neurons[nid]
					if candNeuron == nil {
						continue
					}
					// intra-lobe candidate: require target group > source group
					if !allowedGroupRoute(fromNeuron.GroupID, candNeuron.GroupID, true) {
						continue
					}
					// require candidate to have recent activity to accept rewiring
					if candNeuron.RecentActivity < cfg.GrowthMinActivity {
						continue
					}
					candidates = append(candidates, nid)
				}
				if len(candidates) > 0 {
					t := candidates[rand.Intn(len(candidates))]
					newID := s.From + "->" + t
					newSyn := &Synapse{
						ID:        newID,
						From:      s.From,
						To:        t,
						FromLobe:  s.FromLobe,
						ToLobe:    s.ToLobe,
						Weight:    s.Weight * wScale,
						Plastic:   s.Plastic,
						MinWeight: s.MinWeight,
						MaxWeight: s.MaxWeight,
					}
					newSyn.LastUsed = now
					newSyn.Age = 0
					newSyn.Cooldown = cfg.Cooldown
					l.AddSynapse(newSyn)
					l.RemoveSynapse(sid)
					rewires++
					fmt.Println("Rewired intra-lobe synapse ", sid, " to ", newID)
				}
			}
		}

		// Growth (naive O(n^2) per lobe): create synapse if pre/post recent activity correlates
		neuronIDs := make([]string, 0, len(l.Neurons))
		for nid := range l.Neurons {
			neuronIDs = append(neuronIDs, nid)
		}
		for i := 0; i < len(neuronIDs); i++ {
			for j := 0; j < len(neuronIDs); j++ {
				if i == j {
					continue
				}
				preid := neuronIDs[i]
				postid := neuronIDs[j]
				// skip existing
				exists := false
				for _, ss := range l.Synapses {
					if ss.From == preid && ss.To == postid {
						exists = true
						break
					}
				}
				if exists {
					continue
				}
				pre := l.Neurons[preid]
				post := l.Neurons[postid]
				corr := pre.RecentActivity * post.RecentActivity
				// require recent activity on both neurons before considering growth
				if pre.RecentActivity < growthMinActivity || post.RecentActivity < growthMinActivity {
					continue
				}
				// enforce group constraint for growth: if pre has a non-zero GroupID, do not grow to a post with same GroupID
				// intra-lobe growth: require strictly increasing group id
				if !allowedGroupRoute(pre.GroupID, post.GroupID, true) {
					continue
				}
				if corr > growthCorrThreshold && rand.Float64() < growthProb {
					syn := &Synapse{
						ID:        preid + "->" + postid,
						From:      preid,
						To:        postid,
						FromLobe:  l.Name,
						ToLobe:    l.Name,
						Weight:    0.1,
						Plastic:   true,
						MinWeight: -1.0,
						MaxWeight: 1.0,
					}
					syn.LastUsed = now
					syn.Age = 0
					l.AddSynapse(syn)
					fmt.Println("Grew new intra-lobe synapse ", syn.ID)
				}
			}
		}
	}

	// Inter-lobe maintenance and growth (stored in b.Synapses)
	for sid, s := range b.Synapses {
		if rewires >= maintenanceLimit {
			break
		}
		if s.Cooldown > 0 {
			continue
		}
		inactivity := now - s.LastUsed
		mobility := 0.0
		if inactivity > inactivityThreshold {
			mobility += 1.0
		}
		if s.Eligibility < eligibilityThreshold {
			mobility += 1.0
		}
		if math.Abs(s.Weight) < 0.01 {
			mobility += 1.0
		}

		p := pMigrateBase * mobility
		if rand.Float64() < p {
			// candidate targets must remain in the same target lobe (respect inter-lobe constraint)
			if toL, ok := b.Lobes[s.ToLobe]; ok {
				candidates := make([]string, 0)
				for nid := range toL.Neurons {
					if nid == s.To {
						continue
					}
					// check exists in global synapses map
					exists := false
					for _, ss := range b.Synapses {
						if ss.From == s.From && ss.To == nid && ss.FromLobe == s.FromLobe && ss.ToLobe == s.ToLobe {
							exists = true
							break
						}

						// record the tick when maintenance completed so the interval can be enforced
						b.lastMaintTick = now
					}
					if !exists {
						candidates = append(candidates, nid)
					}
				}
				if len(candidates) > 0 {
					// enforce group constraint using the source neuron from its source lobe
					fromL, okFromL := b.Lobes[s.FromLobe]
					var fromNeuron *Neuron
					if okFromL {
						fromNeuron = fromL.Neurons[s.From]
					}
					// filter candidates by group constraint using allowedGroupRoute
					filtered := make([]string, 0, len(candidates))
					for _, nid := range candidates {
						cand := toL.Neurons[nid]
						if cand == nil {
							continue
						}
						if fromNeuron != nil {
							// inter-lobe migration: allowed regardless of group id
							if !allowedGroupRoute(fromNeuron.GroupID, cand.GroupID, false) {
								continue
							}
						}
						// require candidate to have recent activity before migrating inter-lobe
						if cand.RecentActivity < cfg.GrowthMinActivity {
							continue
						}
						filtered = append(filtered, nid)
					}
					if len(filtered) == 0 {
						continue
					}
					t := filtered[rand.Intn(len(filtered))]
					newID := s.From + "_to_" + t
					newSyn := &Synapse{
						ID:        newID,
						From:      s.From,
						To:        t,
						FromLobe:  s.FromLobe,
						ToLobe:    s.ToLobe,
						Weight:    s.Weight * wScale,
						Plastic:   s.Plastic,
						MinWeight: s.MinWeight,
						MaxWeight: s.MaxWeight,
					}
					newSyn.LastUsed = now
					newSyn.Age = 0
					newSyn.Cooldown = cfg.Cooldown
					b.AddSynapse(newSyn)
					b.RemoveSynapse(sid)
					rewires++
					fmt.Println("Rewired inter-lobe synapse ", sid, " to ", newID)
				}
			}
		}

		// growth across lobes: check all neuron pairs in the respective lobes
		fromL, ok1 := b.Lobes[s.FromLobe]
		toL, ok2 := b.Lobes[s.ToLobe]
		if ok1 && ok2 {
			for preid, pre := range fromL.Neurons {
				for toid, post := range toL.Neurons {
					// skip existing
					exists := false
					for _, ss := range b.Synapses {
						if ss.From == preid && ss.To == toid && ss.FromLobe == fromL.Name && ss.ToLobe == toL.Name {
							exists = true
							break
						}
					}
					if exists {
						continue
					}
					corr := pre.RecentActivity * post.RecentActivity
					// require recent activity on both neurons before considering growth
					if pre.RecentActivity < growthMinActivity || post.RecentActivity < growthMinActivity {
						continue
					}
					// inter-lobe growth: allowed regardless of group id (lobes differ)
					if !allowedGroupRoute(pre.GroupID, post.GroupID, false) {
						continue
					}
					if corr > growthCorrThreshold && rand.Float64() < growthProb {
						syn := &Synapse{
							ID:        fromL.Name + "_" + preid + "_to_" + toL.Name + "_" + toid,
							From:      preid,
							To:        toid,
							FromLobe:  fromL.Name,
							ToLobe:    toL.Name,
							Weight:    0.1,
							Plastic:   true,
							MinWeight: -1.0,
							MaxWeight: 1.0,
						}
						syn.LastUsed = now
						syn.Age = 0
						b.AddSynapse(syn)
						fmt.Println("Grew new inter-lobe synapse between ", fromL.Name, ":", preid, " and ", toL.Name, ":", toid)
					}
				}
			}
		}
	}
}

// allowedGroupRoute returns true if a synapse may be formed from a neuron in
// group `from` to group `to`. The semantics depend on whether the source and
// target are in the same lobe (sameLobe == true) or different lobes
// (sameLobe == false):
//   - sameLobe == true: allowed only when to.GroupID > from.GroupID (strictly
//     increasing), which prevents loops within a lobe.
//   - sameLobe == false: allowed for any group IDs (inter-lobe connections are
//     permitted between any neurons in different lobes).
func allowedGroupRoute(from, to int, sameLobe bool) bool {
	if !sameLobe {
		return true
	}
	return to > from
}

func (b *Brain) AddLobe(lobe *Lobe) {
	b.Lobes[lobe.Name] = lobe
	b.updateLobeOrder()
}

func (b *Brain) GetLobe(name string) *Lobe {
	if lobe, ok := b.Lobes[name]; ok {
		return lobe
	}
	return nil
}

func (b *Brain) AddSynapse(synapse *Synapse) {
	// If this is an intra-lobe synapse, also add it to the lobe's synapse map
	if synapse.FromLobe != "" && synapse.FromLobe == synapse.ToLobe {
		if l, ok := b.Lobes[synapse.FromLobe]; ok {
			l.AddSynapse(synapse)
		}
	} else {
		b.Synapses[synapse.ID] = synapse
	}
	b.updateLobeOrder()
}

func (b *Brain) GetSynapse(id string) *Synapse {
	if synapse, ok := b.Synapses[id]; ok {
		return synapse
	}
	return nil
}

func (b *Brain) RemoveSynapse(id string) {
	delete(b.Synapses, id)
	// Also remove from lobe if it's an intra-lobe synapse
	for _, lobe := range b.Lobes {
		if _, ok := lobe.Synapses[id]; ok {
			lobe.RemoveSynapse(id)
		}
	}
	b.updateLobeOrder()
}

func (b *Brain) Connect(fromLobe, fromNeuron, toLobe, toNeuron string, weight float64, plastic bool) *Synapse {
	synapse := &Synapse{
		ID:        fromLobe + "_" + fromNeuron + "_to_" + toLobe + "_" + toNeuron,
		From:      fromNeuron,
		To:        toNeuron,
		FromLobe:  fromLobe,
		ToLobe:    toLobe,
		Weight:    weight,
		Plastic:   plastic,
		MinWeight: -1.0,
		MaxWeight: 1.0,
	}
	b.AddSynapse(synapse)

	return synapse
}

func (b *Brain) updateLobeOrder() {
	// simple topological sort based on synapse connections
	visited := make(map[string]bool)
	order := []string{}

	var visit func(string)
	visit = func(lobeName string) {
		if visited[lobeName] {
			return
		}
		visited[lobeName] = true

		// visit all lobes that this lobe connects to
		for _, syn := range b.Synapses {
			if syn.FromLobe == lobeName {
				visit(syn.ToLobe)
			}
		}

		order = append(order, lobeName)
	}

	for lobeName := range b.Lobes {
		visit(lobeName)
	}

	// reverse order so that sources (lobes with no incoming edges) come before targets
	for i, j := 0, len(order)-1; i < j; i, j = i+1, j-1 {
		order[i], order[j] = order[j], order[i]
	}
	b.updateOrder = order
}

func LoadFromJson(data []byte) (*Brain, error) {
	var brain Brain
	if err := json.Unmarshal(data, &brain); err != nil {
		return nil, err
	}

	// Initialize maps if nil
	if brain.Lobes == nil {
		brain.Lobes = make(map[string]*Lobe)
	}
	if brain.Synapses == nil {
		brain.Synapses = make(map[string]*Synapse)
	}

	brain.updateLobeOrder()
	for _, lobe := range brain.Lobes {
		lobe.updateNeuronOrder()
	}

	brain.MaintConfig = MaintenanceConfig{
		MaintenanceLimit:     2,
		WScale:               0.05,
		PMigrateBase:         0.02,
		InactivityThreshold:  20000,
		EligibilityThreshold: 0.005,
		GrowthCorrThreshold:  0.9,
		GrowthProb:           0.002,
		GrowthMinActivity:    0.75,
		Cooldown:             2000,
		// By default, run maintenance every 100 ticks to avoid excessive
		// structural changes when the simulation is stepped every frame.
		MaintInterval: 1000,
	}
	return &brain, nil
}

func LoadBrain(filePath string) (*Brain, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}
	return LoadFromJson(data)
}

func (b *Brain) Save(filePath string) error {
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(filePath, data, 0644)
}

func (b *Brain) ToJSON() ([]byte, error) {
	return json.MarshalIndent(b, "", "  ")
}
