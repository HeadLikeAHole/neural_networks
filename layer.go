package main

import (
	"gonum.org/v1/gonum/mat"
)

type LayerDense struct {
	inputs  *mat.Dense
	weights *mat.Dense
	biases  []float64
	Outputs *mat.Dense

	// partial derivatives
	dInputs  *mat.Dense
	dWeights *mat.Dense
	dBiases  []float64
}

func NewLayerDense(nInputs, nNeurons int) *LayerDense {
	l := new(LayerDense)

	l.inputs = new(mat.Dense)

	// nInputs = 2, nNeurons = 3
	// |i1n1 i1n2 i1n3|
	// |i2n1 i2n2 i2n3|
	randWeights := ToDense(Rand2DSlice(-1, 1, nInputs, nNeurons))
	l.weights = new(mat.Dense)
	l.weights.Scale(0.01, randWeights)

	l.biases = make([]float64, nNeurons)

	l.dInputs = new(mat.Dense)
	l.dWeights = new(mat.Dense)
	l.dBiases = make([]float64, nNeurons)

	return l
}

func (l *LayerDense) Forward(inputs *mat.Dense) {
	l.inputs = inputs
	outputs := new(mat.Dense)
	outputs.Mul(inputs, l.weights)

	nRows, nCols := outputs.Dims()
	biasesMat := SliceToMat(nRows, nCols, l.biases)

	outputs.Add(outputs, biasesMat)
	l.Outputs = outputs
}

func (l *LayerDense) Backward(dValues *mat.Dense) {
	// todo: not sure if l.weights needs to be transposed
	l.dInputs.Mul(dValues, l.weights)
	// todo: args should swap places probably
	l.dWeights.Mul(dValues, l.dInputs)
	l.dBiases = SumCols(dValues)
}
