package ann

import "log/slog"

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

func computeLoss(predicted, actual []float64) float64 {
	if len(predicted) != len(actual) {
		return 0.0
	}
	var sum float64
	for i := range predicted {
		diff := predicted[i] - actual[i]
		sum += diff * diff
	}
	return sum / float64(len(predicted))
}
