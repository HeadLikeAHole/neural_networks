package main

import (
	"encoding/json"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"net/http"
)

type JSONRes struct {
	Inputs  [][]float64 `json:"x"`
	Targets []float64   `json:"y"`
}

func main() {
	res, err := http.Get("http://127.0.0.1:8000/")
	if err != nil {
		fmt.Println(err)
	}
	defer res.Body.Close()

	var data JSONRes
	if err = json.NewDecoder(res.Body).Decode(&data); err != nil {
		fmt.Println(err)
	}
	fmt.Println(data)

	dense1 := NewLayerDense(2, 3)
	activation1 := NewActivationReLU()

	dense2 := NewLayerDense(3, 3)
	activation2 := NewActivationSoftmax()

	fmt.Println("init:", data.Inputs)
	fmt.Println()
	dense1.Forward(ToDense(data.Inputs))
	fmt.Println("dense1:", mat.Formatted(dense1.Outputs))
	fmt.Println()
	activation1.Forward(dense1.Outputs)
	fmt.Println("activation1:", mat.Formatted(activation1.Outputs))
	fmt.Println()

	dense2.Forward(activation1.Outputs)
	fmt.Println("dense2:", mat.Formatted(dense2.Outputs))
	fmt.Println()
	activation2.Forward(dense2.Outputs)
	fmt.Println("activation2:", mat.Formatted(activation2.Outputs))
	fmt.Println()

	l := NewLoss(&LossCCE{})
	loss := l.Calculate(activation2.Outputs, data.Targets)
	fmt.Println("loss:", loss)

	predictions := MaxInRowIndex(activation2.Outputs)
	fmt.Println("predictions:", predictions)
	fmt.Println("targets:", data.Targets)
	accuracy := Mean(Compare(predictions, ConvertSlice[float64, int](data.Targets)))
	fmt.Println("acc:", accuracy)

	r1, c1 := dense1.weights.Dims()
	d1Weights := mat.NewDense(r1, c1, nil)
	d1Weights.Copy(dense1.weights)
	d1Biases := make([]float64, len(dense1.biases))
	copy(d1Biases, dense1.biases)
	fmt.Println("d1Weights:", mat.Formatted(d1Weights))
	fmt.Println("d1Biases:", d1Biases)

	r2, c2 := dense1.weights.Dims()
	d2Weights := mat.NewDense(r2, c2, nil)
	d2Weights.Copy(dense2.weights)
	d2Biases := make([]float64, len(dense2.biases))
	copy(d2Biases, dense2.biases)
	fmt.Println("d2Weights:", mat.Formatted(d2Weights))
	fmt.Println("d2Biases:", d2Biases)
}
