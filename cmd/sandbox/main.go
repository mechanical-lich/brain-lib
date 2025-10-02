package main

import (
	"log/slog"
)

func main() {
	//logger := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	//slog.SetDefault(logger)
	slog.Info("Sandbox")
}
