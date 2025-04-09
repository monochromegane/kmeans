package kmeans

import (
	"encoding/gob"
	"math"
	"math/rand/v2"
	"sort"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type LinearAlgebraKMeans struct {
	state *LinearAlgebraKMeansState
}

type LinearAlgebraKMeansState struct {
	InitMethod  int
	NumClusters int
	NumFeatures int
	Centroids   *mat.Dense
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
		state: &LinearAlgebraKMeansState{
			InitMethod:  initMethod,
			NumClusters: numClusters,
			NumFeatures: numFeatures,
			Centroids:   mat.NewDense(numClusters, numFeatures, nil),
		},
	}, nil
}

func LoadLinearAlgebraKMeans(dec *gob.Decoder) (*LinearAlgebraKMeans, error) {
	state := &LinearAlgebraKMeansState{}
	err := dec.Decode(state)
	if err != nil {
		return nil, err
	}
	return &LinearAlgebraKMeans{
		state: state,
	}, nil
}

func (km *LinearAlgebraKMeans) Train(data []float64, iter int, tol float64) (int, float64, error) {
	if len(data) == 0 {
		return 0, 0.0, ErrEmptyData
	}
	if len(data)%km.state.NumFeatures != 0 {
		return 0, 0.0, ErrInvalidDataLength
	}
	if km.state.NumClusters > len(data)/km.state.NumFeatures {
		return 0, 0.0, ErrFewerClustersThanData
	}

	N := int(len(data) / km.state.NumFeatures)
	X := mat.NewDense(N, km.state.NumFeatures, data)

	xNorm := normVec(X)
	XX := tile(N, km.state.NumClusters, xNorm)
	dist := mat.NewDense(N, km.state.NumClusters, nil)
	E := mat.NewDense(N, km.state.NumClusters, nil)
	ETE := mat.NewDense(km.state.NumClusters, km.state.NumClusters, nil)
	invETEData := make([]float64, km.state.NumClusters)

	if km.state.InitMethod == INIT_RANDOM {
		km.initializeRandom(X)
	} else if km.state.InitMethod == INIT_KMEANS_PLUS_PLUS {
		km.initializeKMeansPlusPlus(X, xNorm)
	}

	loss := math.Inf(1)
	numIter := 0
	for i := 0; i < iter; i++ {
		squaredEuclideanDistance(X, km.state.Centroids, XX, dist)
		E.Zero()
		newLoss, err := membership(dist, E)
		if err != nil {
			return 0, 0.0, err
		}
		if math.Abs(loss-newLoss) < tol {
			break
		}
		loss = newLoss

		ETE.Mul(E.T(), E)
		for k := 0; k < km.state.NumClusters; k++ {
			invETEData[k] = 1.0 / ETE.At(k, k)
		}
		invETE := mat.NewDiagDense(km.state.NumClusters, invETEData)

		km.state.Centroids.Mul(E.T(), X)
		km.state.Centroids.Mul(invETE, km.state.Centroids)
		numIter = i
	}
	return numIter, loss, nil
}

func (km *LinearAlgebraKMeans) Predict(data []float64, fn func(row, minCol int, minVal float64) error) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	N := int(len(data) / km.state.NumFeatures)
	X := mat.NewDense(N, km.state.NumFeatures, data)

	xNorm := normVec(X)
	XX := tile(N, km.state.NumClusters, xNorm)
	dist := mat.NewDense(N, km.state.NumClusters, nil)

	squaredEuclideanDistance(X, km.state.Centroids, XX, dist)
	return minIndecies(dist, fn)
}

func (km *LinearAlgebraKMeans) Centroids() [][]float64 {
	centroids := make([][]float64, km.state.NumClusters)
	for i := 0; i < km.state.NumClusters; i++ {
		centroids[i] = make([]float64, km.state.NumFeatures)
		for j := 0; j < km.state.NumFeatures; j++ {
			centroids[i][j] = km.state.Centroids.At(i, j)
		}
	}
	return centroids
}

func (km *LinearAlgebraKMeans) Encode(enc *gob.Encoder) error {
	return enc.Encode(km.state)
}

func (km *LinearAlgebraKMeans) Decode(dec *gob.Decoder) error {
	km.state.Centroids = nil
	return dec.Decode(km.state)
}

func (km *LinearAlgebraKMeans) initializeRandom(X *mat.Dense) {
	N, _ := X.Dims()
	indecies := rand.Perm(N)[:km.state.NumClusters]
	for i, idx := range indecies {
		km.state.Centroids.SetRow(i, X.RowView(idx).(*mat.VecDense).RawVector().Data)
	}
}

func (km *LinearAlgebraKMeans) initializeKMeansPlusPlus(X *mat.Dense, xNorm *mat.VecDense) {
	N, _ := X.Dims()
	idx := rand.IntN(N)
	distances := make([]float64, N)
	for n := 0; n < N; n++ {
		distances[n] = math.Inf(1)
	}

	XX := tile(N, 1, xNorm)
	centroidsData := make([]float64, km.state.NumClusters*km.state.NumFeatures)
	latestCentroidData := X.RowView(idx).(*mat.VecDense).RawVector().Data
	copy(centroidsData[0:len(latestCentroidData)], latestCentroidData)
	latestCentroid := mat.NewDense(1, km.state.NumFeatures, latestCentroidData)

	indecies := make([]int, N)
	indecies[0] = idx
	dist := mat.NewDense(N, 1, nil)

	for k := 1; k < km.state.NumClusters; k++ {
		squaredEuclideanDistance(X, latestCentroid, XX, dist)
		minIndecies(dist, func(row, minCol int, minVal float64) error {
			if minVal < distances[row] {
				distances[row] = minVal
			}
			return nil
		})

		cumSumDist := make([]float64, N)
		floats.CumSum(cumSumDist, dist.ColView(0).(*mat.VecDense).RawVector().Data)
		threshold := rand.Float64() * cumSumDist[N-1]
	SAMPLE:
		for {
			idx := sort.Search(N, func(i int) bool {
				return cumSumDist[i] >= threshold
			})
			for j := 0; j < k; j++ {
				if indecies[j] == idx {
					threshold = rand.Float64() * cumSumDist[N-1]
					continue SAMPLE
				}
			}
			indecies[k] = idx
			latestCentroidData = X.RowView(idx).(*mat.VecDense).RawVector().Data
			copy(centroidsData[k*km.state.NumFeatures:(k+1)*km.state.NumFeatures], latestCentroidData)
			latestCentroid = mat.NewDense(1, km.state.NumFeatures, latestCentroidData)
			break
		}
	}
	km.state.Centroids = mat.NewDense(km.state.NumClusters, km.state.NumFeatures, centroidsData)
}

func membership(dist *mat.Dense, E *mat.Dense) (float64, error) {
	loss := 0.0
	err := minIndecies(dist, func(row, minCol int, minVal float64) error {
		E.Set(row, minCol, 1)
		loss += minVal
		return nil
	})
	if err != nil {
		return 0.0, err
	}
	return loss, nil
}

func minIndecies(dist *mat.Dense, fn func(row, minCol int, minVal float64) error) error {
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
		err := fn(i, minIdx, minVal)
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

func squaredEuclideanDistance(X, centroids, XX, XCT *mat.Dense) {
	N, _ := X.Dims()
	K, _ := centroids.Dims()

	cNorm := normVec(centroids)
	CC := tile(K, N, cNorm)

	XCT.Mul(X, centroids.T())
	XCT.Scale(-2, XCT)

	XCT.Add(XCT, XX)
	XCT.Add(XCT, CC.T())
}
