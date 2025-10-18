package brain

type Brain struct {
	Lobes map[string]*Lobe `json:"lobes"`
}

func NewBrain() *Brain {
	b := &Brain{
		Lobes: make(map[string]*Lobe),
	}
	return b
}

func (b *Brain) Step(dt float64) {
	for _, lobe := range b.Lobes {
		lobe.Step(dt)
	}
}

func (b *Brain) AddLobe(lobe *Lobe) {
	b.Lobes[lobe.Name] = lobe
}

func (b *Brain) GetLobe(name string) *Lobe {
	if lobe, ok := b.Lobes[name]; ok {
		return lobe
	}
	return nil
}
