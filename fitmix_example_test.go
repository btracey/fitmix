package fitmix_test

import (
	"fmt"

	"github.com/btracey/fitmix"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distuv"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

func ExampleEM() {
	src := rand.NewSource(0)
	// First, construct a base Gaussian mixture.
	m1 := []float64{10, 20}
	s1 := mat.NewSymDense(2, []float64{1, 0.1, 0.1, 2})
	n1, _ := distmv.NewNormal(m1, s1, src)
	m2 := []float64{-10, -20}
	s2 := mat.NewSymDense(2, []float64{4, 0.2, 0.2, 0.5})
	n2, _ := distmv.NewNormal(m2, s2, src)
	m3 := []float64{-20, 30}
	s3 := mat.NewSymDense(2, []float64{0.5, -0.3, -0.3, 0.4})
	n3, _ := distmv.NewNormal(m3, s3, src)
	compWeights := []float64{0.5, 0.3, 0.2}
	cat := distuv.NewCategorical(compWeights, src)
	dists := []*distmv.Normal{n1, n2, n3}

	// Sample the mixture to generate a dataset.
	nSamples := 5000
	dim := 2
	xs := mat.NewDense(nSamples, dim, nil)
	for i := 0; i < nSamples; i++ {
		idx := int(cat.Rand())
		dists[idx].Rand(xs.RawRowView(i))
	}

	// Fit the Mixture model.
	em := &fitmix.EM{
		ComponentFitter: &fitmix.GaussianFitter{},
		NumComponents:   3,
		Tol:             1e-12,
		Src:             src,
	}
	comps, weights := em.FitMixture(xs)
	mf1 := comps[0].(*distmv.Normal).Mean(nil)
	mf2 := comps[1].(*distmv.Normal).Mean(nil)
	mf3 := comps[2].(*distmv.Normal).Mean(nil)
	sf1 := comps[0].(*distmv.Normal).CovarianceMatrix(nil)
	sf2 := comps[1].(*distmv.Normal).CovarianceMatrix(nil)
	sf3 := comps[2].(*distmv.Normal).CovarianceMatrix(nil)
	_ = comps
	_ = weights

	// Print the discovered components and their weights. Note that an EM algorithm
	// may permute the order of the terms.
	fmt.Println("True Distribution:")
	fmt.Printf("Weights: %v     %v      %v\n", compWeights[0], compWeights[1], compWeights[2])
	fmt.Printf("Means  : %v %v %v\n", m1, m2, m3)
	fmt.Println("Sigma1:", mat.Formatted(s1, mat.Prefix("        ")))
	fmt.Println("Sigma2:", mat.Formatted(s2, mat.Prefix("        ")))
	fmt.Println("Sigma3:", mat.Formatted(s3, mat.Prefix("        ")))
	fmt.Println("")
	fmt.Println("Discovered Distribution")
	fmt.Printf("Weights: %0.2v       %0.2v       %0.2v\n", weights[0], weights[1], weights[2])
	fmt.Printf("Means  : %0.2v %0.2v %0.2v\n", mf1, mf2, mf3)
	fmt.Printf("Sigma1: %0.2v\n", mat.Formatted(sf1, mat.Prefix("        ")))
	fmt.Printf("Sigma2: %0.2v\n", mat.Formatted(sf2, mat.Prefix("        ")))
	fmt.Printf("Sigma3: %0.2v\n", mat.Formatted(sf3, mat.Prefix("        ")))
	// Output:
	// blah
}
