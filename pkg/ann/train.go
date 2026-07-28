package ann

import (
	"encoding/json"
	"log/slog"
	"math"
	"os"
)

type TrainingExample struct {
	Inputs  []float64
	Outputs []float64
}

func Train(ann *ANN, trainingData []TrainingExample, epochs int, learningRate float64) {
	for e := 0; e < epochs; e++ {
		var totalLoss float64
		for _, example := range trainingData {
			output := ann.Process(example.Inputs)
			loss := computeLoss(output, example.Outputs)
			totalLoss += loss
			ann.Backpropagate(example.Inputs, example.Outputs, learningRate)
		}
		if e%100 == 0 {
			slog.Info("Training progress", "epoch", e, "avg_loss", totalLoss/float64(len(trainingData)))
		}
	}
}

// computeLoss returns mean squared error.
//
// A length mismatch returns +Inf rather than 0.0. Returning zero reported
// *perfect* loss for malformed data, so a training run on a bad dataset looked
// like it had converged immediately.
func computeLoss(predicted, actual []float64) float64 {
	if len(predicted) != len(actual) {
		return math.Inf(1)
	}
	var sum float64
	for i := range predicted {
		diff := predicted[i] - actual[i]
		sum += diff * diff
	}
	return sum / float64(len(predicted))
}

func SaveTrainingData(filename string, examples []TrainingExample) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	enc := json.NewEncoder(file)
	enc.SetIndent("", "  ")
	if err := enc.Encode(examples); err != nil {
		return err
	}
	return nil
}
