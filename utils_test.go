package main

import (
	"fmt"
	"gonum.org/v1/gonum/mat"
	"testing"
)

func TestSliceToOneHotMat(t *testing.T) {
	s := []float64{2, 4, 1, 0, 0, 5, 6}
	d := SliceToOneHotMat(s)
	fmt.Println("dense1:", mat.Formatted(d))
}
