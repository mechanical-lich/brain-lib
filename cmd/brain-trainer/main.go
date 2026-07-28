package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/mechanical-lich/brain-lib/pkg/ann"
)

func main() {
	mode := flag.String("mode", "train", "Mode: train or test")
	annFile := flag.String("ann", "ann.json", "ANN file (for save/load)")
	dataFile := flag.String("data", "", "Training data file (JSON)")
	epochs := flag.Int("epochs", 1000, "Training epochs")
	lr := flag.Float64("lr", 0.1, "Learning rate")
	hidden := flag.Int("hidden", 1, "Number of hidden layers")
	hiddenSize := flag.Int("hiddenSize", 0, "Neurons per hidden layer (default: numInputs+numOutputs)")
	flag.Parse()

	switch *mode {
	case "train":
		if *dataFile == "" {
			fmt.Println("Please provide -data training file.")
			os.Exit(1)
		}
		// Load training data
		f, err := os.Open(*dataFile)
		if err != nil {
			fmt.Println("Error opening data file:", err)
			os.Exit(1)
		}
		defer f.Close()
		var training []ann.TrainingExample
		if err := json.NewDecoder(f).Decode(&training); err != nil {
			fmt.Println("Error decoding training data:", err)
			os.Exit(1)
		}
		// Infer network shape from first example
		numInputs := len(training[0].Inputs)
		numOutputs := len(training[0].Outputs)
		size := *hiddenSize
		if size <= 0 {
			size = numInputs + numOutputs
		}
		var net *ann.ANN
		// Try to load existing network
		if _, err := os.Stat(*annFile); err == nil {
			net, err = ann.LoadANN(*annFile)
			if err != nil {
				fmt.Println("Error loading existing ANN:", err)
				os.Exit(1)
			}
			fmt.Println("Loaded existing ANN from", *annFile)
		} else {
			net = ann.NewANN(numInputs, numOutputs, *hidden, size)
			fmt.Println("Created new ANN")
		}
		ann.Train(net, training, *epochs, *lr)
		// Save network
		if err := net.Save(*annFile); err != nil {
			fmt.Println("Error saving ANN:", err)
			os.Exit(1)
		}
		fmt.Println("Training complete. Saved to", *annFile)

	case "test":
		net, err := ann.LoadANN(*annFile)
		if err != nil {
			fmt.Println("Error loading ANN:", err)
			os.Exit(1)
		}
		reader := bufio.NewReader(os.Stdin)
		fmt.Printf("Loaded ANN. Enter comma-separated inputs (length %d):\n", net.NumInputs)
		for {
			fmt.Print("> ")
			line, err := reader.ReadString('\n')
			if err != nil {
				fmt.Println("Error reading input:", err)
				continue
			}
			line = strings.TrimSpace(line)
			if line == "" || line == "exit" {
				break
			}
			parts := strings.Split(line, ",")
			if len(parts) != net.NumInputs {
				fmt.Printf("Expected %d inputs, got %d\n", net.NumInputs, len(parts))
				continue
			}
			inputs := make([]float64, net.NumInputs)
			bad := false
			for i, p := range parts {
				v, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
				if err != nil {
					// `continue` here used to skip only this field, leaving it
					// at 0 and running the net on silently corrupted input.
					fmt.Printf("Invalid input '%s'\n", p)
					bad = true
					break
				}
				inputs[i] = v
			}
			if bad {
				continue
			}
			out := net.Process(inputs)
			fmt.Printf("Output: %v\n", out)
		}
	default:
		fmt.Println("Unknown mode. Use -mode train or test.")
	}
}
