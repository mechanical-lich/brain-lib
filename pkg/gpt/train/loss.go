// Package train contains everything needed to fit a model.GPT: the loss,
// an AdamW optimizer with gradient clipping and a learning-rate schedule,
// checkpointing, and a generic training loop driven by a Batcher.
//
// Nothing here knows about text. A Batcher just yields (input, target) id
// sequences, so the same loop trains on prose or on sensor/action tokens.
package train

import (
	"fmt"
	"math"

	"github.com/mechanical-lich/brain-lib/pkg/gpt/tensor"
)

// CrossEntropy computes the mean negative log-likelihood of targets under
// softmax(logits), for logits [N, V] and len(targets) == N. It returns the
// loss and dLogits, the gradient of the mean loss with respect to logits,
// ready to pass to GPT.Backward.
func CrossEntropy(logits *tensor.Mat, targets []int) (float64, *tensor.Mat) {
	n, v := logits.Rows, logits.Cols
	if len(targets) != n {
		panic(fmt.Sprintf("train: %d targets for %d logit rows", len(targets), n))
	}
	dLogits := tensor.New(n, v)
	inv := 1 / float64(n)
	var total float64
	for i := 0; i < n; i++ {
		row := logits.Row(i)
		target := targets[i]
		if target < 0 || target >= v {
			panic(fmt.Sprintf("train: target %d out of range [0, %d)", target, v))
		}
		maxv := math.Inf(-1)
		for _, x := range row {
			if x > maxv {
				maxv = x
			}
		}
		var sum float64
		for _, x := range row {
			sum += math.Exp(x - maxv)
		}
		logSum := math.Log(sum) + maxv
		total += logSum - row[target]
		// d/dlogit_j of (-log softmax_target) = softmax_j - [j == target]
		drow := dLogits.Row(i)
		for j, x := range row {
			drow[j] = math.Exp(x-logSum) * inv
		}
		drow[target] -= inv
	}
	return total * inv, dLogits
}
