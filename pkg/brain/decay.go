package brain

import "math"

// decayFactor returns the multiplier for exponential decay at `rate` per second
// applied over `dt` seconds.
//
// This replaces the linear form `x -= x * rate * dt` used throughout the
// library. That form is a first-order Euler step: it is only accurate while
// `rate*dt` is small, it undershoots in the middle of the range, and above
// `rate*dt == 2` it diverges outright — a modulator with DecayRate 3 at dt 1.0
// went 1 -> -2 -> 4 -> -8 -> 16, and a runaway dopamine level slams every
// plastic weight to its clip.
//
// math.Exp(-rate*dt) is the exact solution at any step size, so it needs no
// clamp and gives identical results whatever the frame rate.
func decayFactor(rate, dt float64) float64 {
	if rate <= 0 || dt <= 0 {
		return 1.0
	}
	return math.Exp(-rate * dt)
}
