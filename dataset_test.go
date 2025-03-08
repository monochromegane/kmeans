package kmeans

import (
	"fmt"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

func generate4ClusterDataset(dim, numSamples int) ([]float64, error) {
	centers := make([]*mat.VecDense, 4)
	for i := 0; i < 4; i++ {
		v := make([]float64, dim)
		for j := 0; j < dim; j++ {
			if (i>>uint(j%2))&1 == 1 {
				v[j] = 5.0
			} else {
				v[j] = -5.0
			}
		}
		centers[i] = mat.NewVecDense(dim, v)
	}

	covData := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			if i == j {
				covData[i*dim+j] = 1.0
			} else {
				covData[i*dim+j] = 0.2
			}
		}
	}
	cov := mat.NewSymDense(dim, covData)

	datasets := make([]*mat.Dense, 4)
	for i := 0; i < 4; i++ {
		X, err := multiNorm(centers[i], cov, numSamples)
		if err != nil {
			return nil, err
		}
		datasets[i] = X
	}

	N, D := datasets[0].Dims()
	var X *mat.Dense
	X = datasets[0]
	for _, x := range datasets[1:] {
		n, _ := X.Dims()
		tmp := mat.NewDense(n+N, D, nil)
		tmp.Stack(X, x)
		X = tmp
	}

	return X.RawMatrix().Data, nil
}

func multiNorm(u *mat.VecDense, S *mat.SymDense, numSamples int) (*mat.Dense, error) {
	d, _ := S.Dims()

	z := make([]float64, numSamples*d)
	for i := range z {
		z[i] = rand.NormFloat64()
	}
	Z := mat.NewDense(numSamples, d, z)

	var chol mat.Cholesky
	if ok := chol.Factorize(S); !ok {
		return nil, fmt.Errorf("covariance matrix must be positive defined")
	}

	var L mat.TriDense
	chol.LTo(&L)

	// Y = Z * L^T + μ
	Y := mat.NewDense(numSamples, d, nil)
	Y.Mul(Z, L.T())
	for i := 0; i < numSamples; i++ {
		row := Y.RawRowView(i)
		for j := range row {
			row[j] += u.AtVec(j)
		}
	}

	return Y, nil
}
