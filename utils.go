package main

import (
	"golang.org/x/exp/constraints"
	"gonum.org/v1/gonum/mat"
	"math/rand"
	"slices"
	"time"
)

type Number interface {
	constraints.Float | constraints.Integer
}

func ToDense(data [][]float64) *mat.Dense {
	return mat.NewDense(len(data), len(data[0]), slices.Concat(data...))
}

// SliceToMat converts [1, 3, 2] to [[1, 3, 2], [1, 3, 2], ...]
func SliceToMat(nRows, nCols int, data []float64) *mat.Dense {
	var biasesData []float64
	for i := 0; i < nRows; i++ {
		biasesData = append(biasesData, data...)
	}
	return mat.NewDense(nRows, nCols, biasesData)
}

func RandSlice(min, max float64, size int) []float64 {
	rand.NewSource(time.Now().UnixNano())
	s := make([]float64, size)
	for i := range s {
		s[i] = min + rand.Float64()*(max-min)
	}
	return s
}

func Rand2DSlice(min, max float64, nRows, nCols int) [][]float64 {
	s := make([][]float64, nRows)
	for i := range s {
		s[i] = RandSlice(min, max, nCols)
	}
	return s
}

func SumRows(matrix *mat.Dense) []float64 {
	var rowSums []float64
	nRows, nCols := matrix.Dims()
	for i := 0; i < nRows; i++ {
		var sum float64
		row := matrix.RowView(i)
		for j := 0; j < nCols; j++ {
			sum += row.AtVec(j)
		}
		rowSums = append(rowSums, sum)
	}
	return rowSums
}

func SumCols(matrix *mat.Dense) []float64 {
	var colSums []float64
	nRows, nCols := matrix.Dims()
	for i := 0; i < nCols; i++ {
		var sum float64
		col := matrix.ColView(i)
		for j := 0; j < nRows; j++ {
			sum += col.AtVec(j)
		}
		colSums = append(colSums, sum)
	}
	return colSums
}

func MaxInRow(matrix *mat.Dense) []float64 {
	var maxVals []float64
	nRows, nCols := matrix.Dims()
	for i := 0; i < nRows; i++ {
		row := matrix.RowView(i)
		mx := row.AtVec(0)
		for j := 1; j < nCols; j++ {
			mx = max(mx, row.AtVec(j))
		}
		maxVals = append(maxVals, mx)
	}
	return maxVals
}

func MaxInRowIndex(matrix *mat.Dense) []int {
	var maxIndices []int
	nRows, nCols := matrix.Dims()
	for i := 0; i < nRows; i++ {
		row := matrix.RowView(i)
		maxIndex := 0
		mx := row.AtVec(0)
		for j := 1; j < nCols; j++ {
			if row.AtVec(j) > mx {
				maxIndex = j
				mx = row.AtVec(j)
			}
		}
		maxIndices = append(maxIndices, maxIndex)
	}
	return maxIndices
}

func MaxInCol(matrix *mat.Dense) []float64 {
	var maxVals []float64
	nRows, nCols := matrix.Dims()
	for i := 0; i < nCols; i++ {
		col := matrix.ColView(i)
		mx := col.AtVec(0)
		for j := 1; j < nRows; j++ {
			mx = max(mx, col.AtVec(j))
		}
		maxVals = append(maxVals, mx)
	}
	return maxVals
}

// SubtractMaxInRow subtracts from each element in each row max row value.
// Modifies passed in matrix.
func SubtractMaxInRow(matrix *mat.Dense, maxVals []float64) {
	nRows, nCols := matrix.Dims()
	for i := 0; i < nRows; i++ {
		mx := maxVals[i]
		for j := 0; j < nCols; j++ {
			val := matrix.At(i, j) - mx
			matrix.Set(i, j, val)
		}
	}
}

// SubtractMaxInCol subtracts from each element in each column max column value.
// Modifies passed in matrix.
func SubtractMaxInCol(matrix *mat.Dense, maxVals []float64) {
	nRows, nCols := matrix.Dims()
	for i := 0; i < nCols; i++ {
		mx := maxVals[i]
		for j := 0; j < nRows; j++ {
			val := matrix.At(j, i) - mx
			matrix.Set(j, i, val)
		}
	}
}

// Pair is a type representing a pair of values
type Pair[T, U any] struct {
	First  T
	Second U
}

// Zip accepts two arrays/slices and zip the values together returning a slice of
// Pairs. If the two arrays/slices are not of equal lengths this function will
// panic.
func Zip[T, U any](left []T, right []U) []Pair[T, U] {
	if len(left) != len(right) {
		panic("cannot zip slices of different lengths")
	}
	pairs := make([]Pair[T, U], 0, len(left))
	for idx, item := range left {
		pairs = append(pairs, Pair[T, U]{
			First:  item,
			Second: right[idx],
		})
	}
	return pairs
}

func Mean[T Number](s []T) float64 {
	if len(s) == 0 {
		return 0
	}
	var sum T
	for _, d := range s {
		sum += d
	}
	return float64(sum) / float64(len(s))
}

// Clip clips values outside the interval to the interval edges.
// For example, if an interval of [0, 1] is specified,
// values smaller than 0 become 0, and values larger than 1 become 1.
func Clip(s []float64, lower, upper float64) []float64 {
	newSlice := make([]float64, len(s))
	for i, val := range s {
		newSlice[i] = max(lower, min(val, upper))
	}
	return newSlice
}

func Compare(s1, s2 []int) []int {
	if len(s1) != len(s2) {
		panic("slices should have the same length")
	}
	var result []int
	for i := 0; i < len(s1); i++ {
		if s1[i] == s2[i] {
			result = append(result, 1)
		} else {
			result = append(result, 0)
		}
	}
	return result
}

func ConvertSlice[T, V Number](s []T) []V {
	newSlice := make([]V, len(s))
	for i := 0; i < len(s); i++ {
		newSlice[i] = V(s[i])
	}
	return newSlice
}

func Convert2DSlice[T, V Number](s [][]T) [][]V {
	newSlice := make([][]V, len(s))
	for i := 0; i < len(s); i++ {
		newSlice[i] = ConvertSlice[T, V](s[i])
	}
	return newSlice
}
