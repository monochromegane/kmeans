package kmeans

import (
	"math"
	"math/rand/v2"
)

type NaiveKMeans struct {
	initMethod  int
	numClusters int
	numFeatures int
	centroids   [][]float64
}

func NewNaiveKMeans(numClusters, numFeatures, initMethod int) (*NaiveKMeans, error) {
	if numClusters <= 0 {
		return nil, ErrInvalidNumClusters
	}
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	if initMethod != INIT_NONE && initMethod != INIT_KMEANS_PLUS_PLUS && initMethod != INIT_RANDOM {
		return nil, ErrInvalidInitMethod
	}

	return &NaiveKMeans{
		initMethod:  initMethod,
		numClusters: numClusters,
		numFeatures: numFeatures,
		centroids:   make([][]float64, numClusters),
	}, nil
}

func (km *NaiveKMeans) Train(data []float64, iter int, tol float64) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	if km.initMethod == INIT_RANDOM {
		km.initializeRandom(data)
	}

	loss := 0.0
	N := int(len(data) / km.numFeatures)
	for i := 0; i < iter; i++ {
		counts := make([]int, km.numClusters)
		newCentroids := make([][]float64, km.numClusters)
		for k := 0; k < km.numClusters; k++ {
			newCentroids[k] = make([]float64, km.numFeatures)
			for d := 0; d < km.numFeatures; d++ {
				newCentroids[k][d] = 0
			}
		}
		for n := 0; n < N; n++ {
			x := data[n*km.numFeatures : (n+1)*km.numFeatures]

			err := naiveMinIndecies(x, km.centroids, func(minCol int) error {
				for d := 0; d < km.numFeatures; d++ {
					newCentroids[minCol][d] += x[d]
				}
				counts[minCol]++
				return nil
			})
			if err != nil {
				return err
			}
		}
		for k := 0; k < km.numClusters; k++ {
			if counts[k] == 0 {
				continue
			}
			for d := 0; d < km.numFeatures; d++ {
				newCentroids[k][d] /= float64(counts[k])
			}
		}

		km.centroids = newCentroids

		// Calculate loss
		newLoss := 0.0
		for n := 0; n < N; n++ {
			x := data[n*km.numFeatures : (n+1)*km.numFeatures]

			err := naiveMinIndecies(x, km.centroids, func(minCol int) error {
				newLoss += naiveSquaredEuclideanDistance(x, km.centroids[minCol])
				return nil
			})
			if err != nil {
				return err
			}
		}
		if math.Abs(loss-newLoss) < tol {
			break
		}
		loss = newLoss
	}

	return nil
}

func (km *NaiveKMeans) Predict(data []float64, fn func(row, minCol int) error) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	N := int(len(data) / km.numFeatures)
	for n := 0; n < N; n++ {
		x := data[n*km.numFeatures : (n+1)*km.numFeatures]

		err := naiveMinIndecies(x, km.centroids, func(minCol int) error {
			return fn(n, minCol)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (km *NaiveKMeans) Centroids() [][]float64 {
	viewCentroids := make([][]float64, len(km.centroids))
	for i, centroid := range km.centroids {
		viewCentroids[i] = make([]float64, len(centroid))
		copy(viewCentroids[i], centroid)
	}
	return viewCentroids
}

func (km *NaiveKMeans) initializeRandom(data []float64) {
	N := int(len(data) / km.numFeatures)
	indecies := rand.Perm(N)[:km.numClusters]
	for i, idx := range indecies {
		km.centroids[i] = data[idx*km.numFeatures : (idx+1)*km.numFeatures]
	}
}

func naiveMinIndecies(x []float64, centroids [][]float64, fn func(minCol int) error) error {
	numClusters := len(centroids)
	minIdx := 0
	minVal := naiveSquaredEuclideanDistance(x, centroids[0])
	for k := 1; k < numClusters; k++ {
		val := naiveSquaredEuclideanDistance(x, centroids[k])
		if val < minVal {
			minVal = val
			minIdx = k
		}
	}
	return fn(minIdx)
}

func naiveSquaredEuclideanDistance(x, y []float64) float64 {
	sum := 0.0
	for i := 0; i < len(x); i++ {
		diff := x[i] - y[i]
		sum += math.Pow(diff, 2)
	}
	return sum
}
