package ml

import "github.com/keijiyoshida/govector/vector"

// Perceptron represents a perceptron.
type Perceptron struct {
	W         []float64
	NumErrors []int
	eta       float64
	numIter   int
}

// Fit trains the perceptron by using the input training data.
func (p *Perceptron) Fit(x [][]float64, y []float64) {
	p.W = make([]float64, len(x[0])+1)
	p.NumErrors = make([]int, p.numIter)

	for i := 0; i < p.numIter; i++ {
		numError := 0

		for j := 0; j < len(x); j++ {
			d := p.eta * (y[i] - p.Predict(x[i]))

			p.W[0] += d

			for k := 0; k < len(x[0]); k++ {
				p.W[k+1] += d * x[i]
			}

			if d != 0 {
				numError++
			}
		}

		p.NumErrors[i] = numError
	}
}

// Predict predicts a labeled value and returns it.
func (p *Perceptron) Predict(x []float64) int {
	if vector.DotProduct(x, p.W[1:])+p.W[0] >= 0.0 {
		return 1
	} else {
		return -1
	}
}
