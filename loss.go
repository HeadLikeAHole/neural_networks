package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type Passer interface {
	Forward(*mat.Dense, any) []float64
	Backward(*mat.Dense, any) *mat.Dense
}

type Loss struct {
	Passer
}

func NewLoss(p Passer) *Loss {
	return &Loss{Passer: p}
}

func (l *Loss) Calculate(inputs *mat.Dense, targets any) float64 {
	sampleLosses := l.Forward(inputs, targets)
	return Mean(sampleLosses)
}

// LossCCE is categorical cross-entropy loss
type LossCCE struct{}

func (l *LossCCE) Forward(inputs *mat.Dense, targets any) []float64 {
	var losses []float64

	// clip data to prevent division by 0
	// clip both sides to not drag mean towards any value
	inputs.Apply(func(i, j int, v float64) float64 {
		return max(1e-7, min(v, 1-1e-7))
	}, inputs)

	// todo: refactor to a single type later if possible
	switch v := targets.(type) {
	case []float64:
		nRows, _ := inputs.Dims()
		for i := 0; i < nRows; i++ {
			row := inputs.RawRowView(i)
			losses = append(losses, -math.Log(row[int(v[i])]))
		}
	case [][]float64:
		targetsMat := ToDense(v)
		inputs.MulElem(inputs, targetsMat)
		confidences := SumRows(inputs)
		for _, c := range confidences {
			losses = append(losses, -math.Log(c))
		}
	}

	return losses
}

func (l *LossCCE) Backward(inputs *mat.Dense, targets any) *mat.Dense {
	var dInputs *mat.Dense
	// dL/dy-hat = -y/y-hat = -target/predicted
	switch v := targets.(type) {
	case []float64:
		targetsMat := SliceToOneHotMat(v)
		// calculate gradient
		dInputs.DivElem(targetsMat, inputs)
		dInputs.Scale(-1, dInputs)
	case [][]float64:
		targetsMat := ToDense(v)
		// calculate gradient
		dInputs.DivElem(targetsMat, inputs)
		dInputs.Scale(-1, dInputs)
	}
	// normalize gradient
	nRows, _ := inputs.Dims()
	// nRows is number of samples. We divide inputs by it.
	dInputs.Scale(float64(1/nRows), dInputs)
	return dInputs
}
