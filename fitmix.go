// package fitmix is a package for fitting mixture models to data.
package fitmix

import (
	"fmt"
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"gonum.org/v1/gonum/stat/distmv"
)

// ComponentFitter is a type that fits a single component of a mixture distribution
// to weighted data. The third index specifies which component is being fit, to
// allow the implementing type to reuse memory or have different types of mixture
// components.
type ComponentFitter interface {
	Init(nSamples, dim, nComp int)
	FitComponent(x mat.Matrix, weights []float64, idx int) distmv.LogProber
}

// EM uses the expectation-maximization algorithm to fit a mixture distribution
// to a set of weighted data.
type EM struct {
	// Fitter fits a single component to weighted data. Must not be nil.
	ComponentFitter ComponentFitter
	// NumComponents sets the number of mixture components. If components is 0,
	// a default of log(nSamples) is used with a minimum of 2 components.
	NumComponents int
	// Tol sets the tolerance for the convergence of the weight update matrix.
	Tol float64
	// MaxIter sets the maximum number of iterations.
	MaxIter int
	// Src specifies the source for random initialization. If nil, the default
	// in exp/rand is used.
	Src rand.Source
	// InitialWeights sets an initial estimate for the weights of the components.
	// If nil, a random initialization would be used.
	InitialWeights mat.Matrix
}

// FitMixture fits a mixture distribution to the input data.
//
// The returned components can be changed to their types with type assertions.
// The returned type will be consistent with the returned type from ComponentFitter.
func (em *EM) FitMixture(xs mat.Matrix) (components []distmv.LogProber, weights []float64) {
	// TODO(btracey): Think if this can/needs to support weighted data.
	// TODO(btracey): Allow parallelism

	// Check all of the inputs.
	r, c := xs.Dims()
	if em.ComponentFitter == nil {
		panic("em: component fitter not set")
	}
	nComp := em.NumComponents
	if nComp == 0 {
		nComp = int(math.Log(float64(nComp)))
		if nComp < 2 {
			nComp = 2
		}
	}
	compWeights := mat.NewDense(r, nComp, nil)
	if em.InitialWeights != nil {
		ri, ci := em.InitialWeights.Dims()
		if ri != r {
			panic("em: initial weight matrix has wrong number of rows")
		}
		if ci != nComp {
			panic("em: initial weight matrix columns does not match number of components")
		}
		compWeights.Copy(em.InitialWeights)
	} else {
		rnd := rand.Float64
		if em.Src != nil {
			rnd = rand.New(em.Src).Float64
		}
		for i := 0; i < r; i++ {
			for j := 0; j < nComp; j++ {
				compWeights.Set(i, j, rnd())
			}
		}
	}

	tol := em.Tol
	if tol == 0 {
		tol = 1e-8
	}

	components = make([]distmv.LogProber, nComp)
	weights = make([]float64, nComp)
	updatedWeights := mat.NewDense(r, nComp, nil)

	em.ComponentFitter.Init(r, c, nComp)
	iter := -1
	x := make([]float64, c)
	tmpWeights := make([]float64, r)
	for {
		iter++
		if em.MaxIter != 0 && iter >= em.MaxIter {
			break
		}
		// Fit all of the components.
		for comp := 0; comp < nComp; comp++ {
			mat.Col(tmpWeights, comp, compWeights)
			for i, v := range tmpWeights {
				tmpWeights[i] = math.Exp(v) // Scale the weights out of log space.
			}
			components[comp] = em.ComponentFitter.FitComponent(xs, tmpWeights, comp)
		}

		// Reassign the per-point weights.
		for i := 0; i < r; i++ {
			mat.Row(x, i, xs)
			w := updatedWeights.RawRowView(i)
			for j := 0; j < nComp; j++ {
				lp := components[j].LogProb(x)
				w[j] = lp
			}
			// Normalize the weights.
			lse := floats.LogSumExp(w)
			for j := range w {
				w[j] -= lse
			}
		}

		// Set the component weights. The above is normalized for each row to
		// sum to 1, so the whole matrix has to sum to r
		for j := 0; j < nComp; j++ {
			mat.Col(tmpWeights, j, updatedWeights)
			weights[j] = math.Exp(floats.LogSumExp(tmpWeights)) / float64(r)
		}
		fmt.Println("weights", weights)

		// Test convergence.
		maxDiff := math.Inf(-1)
		for i := 0; i < r; i++ {
			for j := 0; j < nComp; j++ {
				lp1 := updatedWeights.At(i, j)
				lp2 := compWeights.At(i, j)
				diff := math.Abs(math.Exp(lp1) - math.Exp(lp2))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
		}
		fmt.Println("maxDiff = ", maxDiff)
		if maxDiff < tol {
			break
		}
		compWeights.Copy(updatedWeights)
	}
	return components, weights
}

type GaussianFitter struct {
	// TODO(btracey): Allow a prior to regularize?

	means  *mat.Dense
	sigmas []*mat.SymDense
	col    [][]float64 // slice of slice for parallelism.
}

func (gf *GaussianFitter) Init(nSamples, dim, nComp int) {
	gf.means = mat.NewDense(nComp, dim, nil)
	gf.sigmas = make([]*mat.SymDense, nComp)
	for i := range gf.sigmas {
		gf.sigmas[i] = mat.NewSymDense(dim, nil)
	}
	gf.col = make([][]float64, nComp)
	for i := range gf.col {
		gf.col[i] = make([]float64, nSamples)
	}

}

func (gf *GaussianFitter) FitComponent(x mat.Matrix, weights []float64, idx int) distmv.LogProber {
	mu := gf.means.RawRowView(idx)
	sigma := gf.sigmas[idx]
	col := gf.col[idx]
	for j := range mu {
		mat.Col(col, j, x)
		mu[j] = stat.Mean(col, weights)
	}
	stat.CovarianceMatrix(sigma, x, weights)

	n, ok := distmv.NewNormal(mu, sigma, nil)
	if !ok {
		// TODO(btracey): Do something better with this panic.
		panic("gaussian fit: bad normal")
	}
	return n
}
