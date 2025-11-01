name = "critters"

vendor:
	go mod vendor

bin:
	mkdir bin
	cp -r ./assets ./bin/assets
	cp -r ./data ./bin/data

.PHONY: clean
clean:
	rm -rf ./bin

run:
	go run *.go

run-vendored:
	go run *.go -mod=vendor

build: clean bin
	go build -o ./bin/$(name)

build-all: clean build-mac build-windows build-linux
	@echo "Builds for all platforms are complete."

build-mac: bin
	GOOS=darwin GOARCH=arm64 go build -o ./bin/$(name)-mac

build-linux: bin
	GOOS=linux GOARCH=amd64 go build -o ./bin/$(name)-linux

build-windows: bin
	GOOS=windows GOARCH=amd64 go build -o ./bin/$(name)-windows.exe

test:
	go test ./... -v

.PHONY: update-go-deps
update-go-deps:
	@echo ">> updating Go dependencies"
	@for m in $$(go list -mod=readonly -m -f '{{ if and (not .Indirect) (not .Main)}}{{.Path}}{{end}}' all); do \
		go get $$m; \
	done
	go mod tidy
ifneq (,$(wildcard vendor))
	go mod vendor
endif
