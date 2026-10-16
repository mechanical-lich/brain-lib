# Builds every command under cmd/ for the host or for other platforms.
#
#   make build                         host binaries in bin/
#   make build-linux                   linux/amd64 + linux/arm64 in bin/linux-<arch>/
#   make build-all                     every platform in PLATFORMS
#   make build-linux TOOLS=gpt-train   only some tools
#   make build-platform PLATFORM=linux/386
#
# Nothing here uses cgo, so cross-compiling needs only the Go toolchain.

BIN       ?= bin
TOOLS     ?= $(notdir $(wildcard cmd/*))
PLATFORMS ?= linux/amd64 linux/arm64 darwin/arm64 darwin/amd64 windows/amd64 windows/arm64
GOFLAGS_BUILD ?= -trimpath
LDFLAGS   ?= -s -w

# Platform targets are named os-arch (e.g. build-linux-amd64), matching
# the output directory.
PLATFORM_TARGETS = $(addprefix build-,$(subst /,-,$(PLATFORMS)))

.PHONY: build build-all build-linux build-mac build-windows build-platform list-tools clean test vet vendor update-go-deps

build:
	@mkdir -p $(BIN)
	@for t in $(TOOLS); do echo ">> $$t"; go build $(GOFLAGS_BUILD) -o $(BIN)/$$t ./cmd/$$t || exit 1; done

build-all: $(PLATFORM_TARGETS)
	@echo "Builds for all platforms are complete."

build-linux: build-linux-amd64 build-linux-arm64
build-mac: build-darwin-arm64 build-darwin-amd64
build-windows: build-windows-amd64 build-windows-arm64

# Any GOOS/GOARCH pair `go tool dist list` knows, e.g. PLATFORM=linux/386.
build-platform:
	@test -n "$(PLATFORM)" || { echo "usage: make build-platform PLATFORM=<os>/<arch>"; exit 1; }
	@$(MAKE) --no-print-directory build-$(subst /,-,$(PLATFORM))

# build-<os>-<arch>: every tool in TOOLS into $(BIN)/<os>-<arch>/.
build-%:
	@os=$(word 1,$(subst -, ,$*)); arch=$(word 2,$(subst -, ,$*)); \
	[ -n "$$arch" ] || { echo "unknown target $@; want build-<os>-<arch>"; exit 1; }; \
	ext=$$( [ "$$os" = windows ] && echo .exe ); \
	out=$(BIN)/$$os-$$arch; mkdir -p $$out; \
	for t in $(TOOLS); do \
		echo ">> $$os/$$arch $$t"; \
		CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch go build $(GOFLAGS_BUILD) -ldflags '$(LDFLAGS)' -o $$out/$$t$$ext ./cmd/$$t || exit 1; \
	done

list-tools:
	@echo $(TOOLS)

clean:
	rm -rf ./$(BIN)

test:
	go test ./...

vet:
	go vet ./...

vendor:
	go mod vendor

update-go-deps:
	@echo ">> updating Go dependencies"
	@for m in $$(go list -mod=readonly -m -f '{{ if and (not .Indirect) (not .Main)}}{{.Path}}{{end}}' all); do \
		go get $$m; \
	done
	go mod tidy
ifneq (,$(wildcard vendor))
	go mod vendor
endif
