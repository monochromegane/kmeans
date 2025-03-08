package kmeans

import (
	"math"
	"math/rand/v2"

	"gonum.org/v1/gonum/mat"
)

type LinearAlgebraKMeans struct {
	initMethod  int
	numClusters int
	numFeatures int
	centroids   *mat.Dense
}

func NewLinearAlgebraKMeans(numClusters, numFeatures, initMethod int) (*LinearAlgebraKMeans, error) {
	if numClusters <= 0 {
		return nil, ErrInvalidNumClusters
	}
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	if initMethod != INIT_NONE && initMethod != INIT_KMEANS_PLUS_PLUS && initMethod != INIT_RANDOM {
		return nil, ErrInvalidInitMethod
	}

	return &LinearAlgebraKMeans{
		initMethod:  initMethod,
		numClusters: numClusters,
		numFeatures: numFeatures,
		centroids:   mat.NewDense(numClusters, numFeatures, nil),
	}, nil
}

func (km *LinearAlgebraKMeans) Train(data []float64, iter int, tol float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	N := int(len(data) / km.numFeatures)
	X := mat.NewDense(N, km.numFeatures, data)

	if km.initMethod == INIT_RANDOM {
		km.initializeRandom(X)
	}
	xNorm := normVec(X)
	XX := tile(N, km.numClusters, xNorm)
	X_EC := mat.NewDense(N, km.numFeatures, nil)
	X_ECT_X_EC := mat.NewDense(km.numFeatures, km.numFeatures, nil)
	loss := 0.0

	for i := 0; i < iter; i++ {
		dist := squaredEuclideanDistance(X, km.centroids, XX)
		E, err := membership(dist)
		if err != nil {
			return err
		}

		ETE := mat.NewDense(km.numClusters, km.numClusters, nil)
		ETE.Mul(E.T(), E)
		invETE := mat.NewDense(km.numClusters, km.numClusters, nil)
		invETE.Inverse(ETE.DiagView())

		km.centroids.Mul(E.T(), X)
		km.centroids.Mul(invETE, km.centroids)

		// Calculate loss
		X_EC.Mul(E, km.centroids)
		X_EC.Sub(X, X_EC)
		X_ECT_X_EC.Mul(X_EC.T(), X_EC)
		newLoss := mat.Trace(X_ECT_X_EC)
		if math.Abs(loss-newLoss) < tol {
			break
		}
		loss = newLoss
	}
	return nil
}

func (km *LinearAlgebraKMeans) Predict(data []float64, fn func(row, minCol int) error) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	N := int(len(data) / km.numFeatures)
	X := mat.NewDense(N, km.numFeatures, data)

	xNorm := normVec(X)
	XX := tile(N, km.numClusters, xNorm)

	dist := squaredEuclideanDistance(X, km.centroids, XX)
	return minIndecies(dist, fn)
}

func (km *LinearAlgebraKMeans) Centroids() [][]float64 {
	centroids := make([][]float64, km.numClusters)
	for i := 0; i < km.numClusters; i++ {
		centroids[i] = make([]float64, km.numFeatures)
		for j := 0; j < km.numFeatures; j++ {
			centroids[i][j] = km.centroids.At(i, j)
		}
	}
	return centroids
}

func (km *LinearAlgebraKMeans) initializeRandom(X *mat.Dense) {
	N, _ := X.Dims()
	indecies := rand.Perm(N)[:km.numClusters]
	for i, idx := range indecies {
		km.centroids.SetRow(i, X.RowView(idx).(*mat.VecDense).RawVector().Data)
	}
}

func membership(dist *mat.Dense) (*mat.Dense, error) {
	N, K := dist.Dims()
	E := mat.NewDense(N, K, nil)
	err := minIndecies(dist, func(row, minCol int) error {
		E.Set(row, minCol, 1)
		return nil
	})
	if err != nil {
		return nil, err
	}
	return E, nil
}

func minIndecies(dist *mat.Dense, fn func(row, minCol int) error) error {
	N, K := dist.Dims()

	for i := 0; i < N; i++ {
		row := dist.RowView(i).(*mat.VecDense)
		minVal := row.AtVec(0)
		minIdx := 0
		for j := 1; j < K; j++ {
			if val := row.AtVec(j); val < minVal {
				minVal = val
				minIdx = j
			}
		}
		err := fn(i, minIdx)
		if err != nil {
			return err
		}
	}
	return nil
}

func normVec(X *mat.Dense) *mat.VecDense {
	N, _ := X.Dims()
	normVec := mat.NewVecDense(N, nil)
	for i := 0; i < N; i++ {
		normVec.SetVec(i, mat.Norm(X.RowView(i), 2))
	}
	normVec.MulElemVec(normVec, normVec)
	return normVec
}

func tile(r, c int, x *mat.VecDense) *mat.Dense {
	X := mat.NewDense(r, c, nil)
	for i := 0; i < c; i++ {
		X.SetCol(i, x.RawVector().Data)
	}
	return X
}

func squaredEuclideanDistance(X, centroids, XX *mat.Dense) *mat.Dense {
	N, _ := X.Dims()
	K, _ := centroids.Dims()

	cNorm := normVec(centroids)
	CC := tile(K, N, cNorm)

	XCT := mat.NewDense(N, K, nil)
	XCT.Mul(X, centroids.T())
	XCT.Scale(-2, XCT)

	XCT.Add(XCT, XX)
	XCT.Add(XCT, CC.T())

	return XCT
}
