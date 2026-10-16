package gpu

// kernel is one WGSL compute shader with a "main" entry point. binds lists
// its storage bindings in order, 'r' for read-only and 'w' for read_write;
// the uniform parameter block always follows them as the last binding.
//
// Every shader keeps its per-thread state in named scalars and vectors: a
// dynamically indexed local array spills to memory and was measured to cost
// 3x in the spike, so no kernel uses one.
type kernel struct {
	name  string
	binds string
	src   string
}

func (k *kernel) source() string { return wgslCommon + k.src }

// wgslCommon is prepended to every shader.
const wgslCommon = `
// flat turns a 2D workgroup grid back into a linear workgroup number; grids
// are 2D only because one dimension is limited to 65535 workgroups.
fn flat(wid: vec3<u32>, nwg: vec3<u32>) -> u32 { return wid.y * nwg.x + wid.x; }
`

// maxGroups is WebGPU's default maxComputeWorkgroupsPerDimension.
const maxGroups = 65535

// grid returns a workgroup grid of at least n workgroups, split into two
// dimensions when n is above the per-dimension limit. Shaders recover the
// workgroup number with flat() and skip the few extra ones.
func grid(n int) [3]uint32 {
	if n <= maxGroups {
		return [3]uint32{uint32(n), 1, 1}
	}
	y := (n + maxGroups - 1) / maxGroups
	x := (n + y - 1) / y
	return [3]uint32{uint32(x), uint32(y), 1}
}

// elems returns a grid with one thread per element for 256-thread
// workgroups.
func elems(n int) [3]uint32 { return grid((n + 255) / 256) }

func ceilDiv(a, b int) int { return (a + b - 1) / b }
