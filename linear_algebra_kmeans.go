package kmeans

import "gonum.org/v1/gonum/mat"

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

func (km *LinearAlgebraKMeans) Train(data []float64, iter int) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	return nil
}

func (km *LinearAlgebraKMeans) Predict(data []float64, fn func(row, minCol int) bool) error {
	return nil
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
