package main

import (
	"gonum.org/v1/gonum/mat"
	"math"
)

type Forwarder interface {
	Forward(*mat.Dense, any) []float64
}

type Loss struct {
	Forwarder
}

func NewLoss(f Forwarder) *Loss {
	return &Loss{Forwarder: f}
}

func (l *Loss) Calculate(outputs *mat.Dense, targets any) float64 {
	sampleLosses := l.Forward(outputs, targets)
	return Mean(sampleLosses)
}

// LossCCE is categorical cross-entropy loss
type LossCCE struct{}

func (l *LossCCE) Forward(outputs *mat.Dense, targets any) []float64 {
	var losses []float64

	// clip data to prevent division by 0
	// clip both sides to not drag mean towards any value
	outputs.Apply(func(i, j int, v float64) float64 {
		return max(1e-7, min(v, 1-1e-7))
	}, outputs)

	switch v := targets.(type) {
	case []float64:
		nRows, _ := outputs.Dims()
		for i := 0; i < nRows; i++ {
			row := outputs.RawRowView(i)
			losses = append(losses, -math.Log(row[int(v[i])]))
		}
	case [][]float64:
		targetsMat := ToDense(v)
		outputs.MulElem(outputs, targetsMat)
		confidences := SumRows(outputs)
		for _, c := range confidences {
			losses = append(losses, -math.Log(c))
		}
	}

	return losses
}
