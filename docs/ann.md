---
title: ANN
nav_order: 6
permalink: /ann/
---

# Feed-forward network (`pkg/ann`)
{: .no_toc }

1. TOC
{:toc}

`pkg/ann` is a classic fully connected feed-forward network trained by
backpropagation on mean squared error. It is the model behind
[`cmd/brain-trainer`](../cli/#brain-trainer) and is not used by the game.

## Shape

`NewANN(numInputs, numOutputs, numLayers, hiddenSize)` builds `numLayers`
hidden layers of `hiddenSize` neurons each, plus an output layer. Weights
and biases are initialized from `N(0, 0.1)`.

```go
net := ann.NewANN(13, 10, 1, 23)     // 13 inputs, 10 outputs, one hidden layer of 23
out := net.Process(inputs)           // nil if len(inputs) != 13
out, err := net.ProcessChecked(inputs)
```

## Training

```go
examples := []ann.TrainingExample{
    {Inputs: []float64{1, 0, 0.97}, Outputs: []float64{0, 1}},
}
ann.Train(net, examples, 1000, 0.1)   // epochs, learning rate
```

`Train` runs one `Backpropagate` per example per epoch and logs the mean
loss every 100 epochs. A malformed example makes `Backpropagate` return an
error rather than panic mid-pass, and `computeLoss` returns `+Inf` for a
length mismatch so a bad dataset cannot look converged.

## Files

`net.Save(path)` and `ann.LoadANN(path)` read and write JSON with camelCase
keys:

```json
{
  "numInputs": 13,
  "numOutputs": 10,
  "numLayers": 2,
  "weights": [[[...], ...], ...],
  "biases": [[...], ...],
  "currentValues": [[...], ...]
}
```

`weights[layer][neuron][input]`; `currentValues` holds the last forward
pass's activations. `ann.SaveTrainingData(path, examples)` writes an
examples file in the shape `brain-trainer -data` reads:

```json
[
  {"Inputs": [1, 0, 0.97], "Outputs": [0, 1]},
  ...
]
```
