module github.com/mechanical-lich/brain-lib/experiments/gpu-spike

go 1.25.1

require (
	github.com/gogpu/gputypes v0.8.0
	github.com/gogpu/wgpu v0.34.5
	github.com/mechanical-lich/brain-lib v0.0.0
)

require (
	github.com/go-webgpu/goffi v0.6.3 // indirect
	github.com/go-webgpu/webgpu v0.5.5 // indirect
	github.com/gogpu/gpucontext v0.31.3 // indirect
	github.com/gogpu/naga v0.19.0 // indirect
	golang.org/x/sys v0.47.0 // indirect
)

replace github.com/mechanical-lich/brain-lib => ../..
