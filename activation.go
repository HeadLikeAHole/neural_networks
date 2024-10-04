package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// ActivationReLU is rectified linear units activation function
type ActivationReLU struct {
	// todo: check later if this field is necessary
	inputs  *mat.Dense
	Outputs *mat.Dense

	dInputs *mat.Dense
}

func NewActivationReLU() *ActivationReLU {
	return &ActivationReLU{
		inputs:  new(mat.Dense),
		Outputs: new(mat.Dense),
		dInputs: new(mat.Dense),
	}
}

func (a *ActivationReLU) Forward(inputs *mat.Dense) {
	a.inputs = inputs
	a.Outputs.Apply(func(i, j int, v float64) float64 {
		return max(0, v)
	}, inputs)
}

func (a *ActivationReLU) Backward(dValues *mat.Dense) {
	// since derivative of ReLU is 1's and 0's we omit it
	// and apply max(0, v) function directly on dValues to simplify calculations
	a.dInputs.Apply(func(i, j int, v float64) float64 {
		return max(0, v)
	}, dValues)
}

type ActivationSoftmax struct {
	Outputs *mat.Dense
}

func NewActivationSoftmax() *ActivationSoftmax {
	return &ActivationSoftmax{Outputs: new(mat.Dense)}
}

func (a *ActivationSoftmax) Forward(inputs *mat.Dense) {
	SubtractMaxInRow(inputs, MaxInRow(inputs))
	fmt.Println("subbed:", mat.Formatted(inputs))
	fmt.Println()

	expVals := new(mat.Dense)
	expVals.Apply(func(i, j int, v float64) float64 {
		return math.Pow(math.E, v)
	}, inputs)
	fmt.Println("exp:", mat.Formatted(expVals))
	fmt.Println()

	rowSums := SumRows(expVals)
	fmt.Println("row sums:", rowSums)
	fmt.Println()

	a.Outputs.Apply(func(i, j int, v float64) float64 {
		return v / rowSums[j]
	}, expVals)
}

func (a *ActivationSoftmax) Backward(dValues *mat.Dense) *mat.Dense {
	return nil
}
