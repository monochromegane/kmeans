package kmeans

import (
	"math"
	"math/rand/v2"
	"runtime"

	"golang.org/x/sync/errgroup"
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

func (km *NaiveKMeans) Train(data []float64, iter int, tol float64) (int, float64, error) {
	if len(data) == 0 {
		return 0, 0.0, ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return 0, 0.0, ErrInvalidDataLength
	}
	if km.numClusters > len(data)/km.numFeatures {
		return 0, 0.0, ErrFewerClustersThanData
	}

	if km.initMethod == INIT_RANDOM {
		km.initializeRandom(data)
	} else if km.initMethod == INIT_KMEANS_PLUS_PLUS {
		km.initializeGreedyKMeansPlusPlus(data)
	}

	loss := math.Inf(1)
	numIter := 0
	N := int(len(data) / km.numFeatures)

	numWorkers := runtime.NumCPU()
	chunkSize := N / numWorkers
	if chunkSize == 0 {
		chunkSize = 1
		numWorkers = N
	}

	type result struct {
		centroids [][]float64
		counts    []int
		loss      float64
	}

	var eg errgroup.Group
	eg.SetLimit(numWorkers)

	for i := range iter {
		results := make(chan result, numWorkers)
		for w := range numWorkers {
			centroids := km.centroids
			start := w * chunkSize
			end := start + chunkSize
			if w == numWorkers-1 {
				end = N
			}
			eg.Go(func() error {
				counts := make([]int, km.numClusters)
				newCentroids := make([][]float64, km.numClusters)
				for k := range km.numClusters {
					newCentroids[k] = make([]float64, km.numFeatures)
				}
				loss := 0.0
				for n := start; n < end; n++ {
					x := data[n*km.numFeatures : (n+1)*km.numFeatures]

					err := naiveMinIndecies(x, centroids, func(minCol int, minDist float64) error {
						for d := range km.numFeatures {
							newCentroids[minCol][d] += x[d]
						}
						counts[minCol]++
						loss += minDist
						return nil
					})
					if err != nil {
						return err
					}
				}
				results <- result{
					centroids: newCentroids,
					counts:    counts,
					loss:      loss,
				}
				return nil
			})
		}
		if err := eg.Wait(); err != nil {
			return 0, 0.0, err
		}
		close(results)

		counts := make([]int, km.numClusters)
		newCentroids := make([][]float64, km.numClusters)
		for k := range km.numClusters {
			newCentroids[k] = make([]float64, km.numFeatures)
		}
		newLoss := 0.0
		for r := range results {
			for k := range km.numClusters {
				for d := range km.numFeatures {
					newCentroids[k][d] += r.centroids[k][d]
				}
				counts[k] += r.counts[k]
			}
			newLoss += r.loss
		}
		loss = newLoss

		frobNorm := 0.0
		centroidDiff := 0.0
		for k := range km.numClusters {
			if counts[k] == 0 {
				continue
			}
			for d := range km.numFeatures {
				newCentroids[k][d] /= float64(counts[k])

				newCentroid := newCentroids[k][d]
				diff := km.centroids[k][d] - newCentroid
				centroidDiff += diff * diff
				frobNorm += newCentroid * newCentroid
			}
		}

		km.centroids = newCentroids
		numIter = i

		if math.Sqrt(centroidDiff)/(math.Sqrt(frobNorm)) < tol {
			break
		}
	}

	return numIter, loss, nil
}

func (km *NaiveKMeans) Predict(data []float64, fn func(row, minCol int, minVal float64) error) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.numFeatures != 0 {
		return ErrInvalidDataLength
	}

	N := int(len(data) / km.numFeatures)
	for n := range N {
		x := data[n*km.numFeatures : (n+1)*km.numFeatures]

		err := naiveMinIndecies(x, km.centroids, func(minCol int, minDist float64) error {
			return fn(n, minCol, minDist)
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

func (km *NaiveKMeans) initializeGreedyKMeansPlusPlus(data []float64) {
	L := 2 + int(math.Log(float64(km.numClusters)))

	N := int(len(data) / km.numFeatures)
	idx := rand.IntN(N)
	distances := make([]float64, N)
	centroids := [][]float64{
		data[idx*km.numFeatures : (idx+1)*km.numFeatures],
	}
	latestCentroid := make([][]float64, 1)
	latestCentroid[0] = centroids[0]

	cumSumDist := make([]float64, N)
	for n := range N {
		distances[n] = math.Inf(1)
		x := data[n*km.numFeatures : (n+1)*km.numFeatures]
		naiveMinIndeciesEarlyReturn(x, latestCentroid, distances[n], func(minCol int, minDist float64) error {
			if minDist < distances[n] {
				distances[n] = minDist
			}
			if n == 0 {
				cumSumDist[n] = distances[n]
			} else {
				cumSumDist[n] = cumSumDist[n-1] + distances[n]
			}
			return nil
		})
	}

	var bestCumSumDist []float64
	var bestDistances []float64
	for k := 1; k < km.numClusters; k++ {
		bestLoss := math.Inf(1)
		bestSelectedIdx := 0
		for range L {
			threshold := rand.Float64() * cumSumDist[N-1]

			left, right := 0, N-1
			var selectedIdx int
			for left <= right {
				mid := (left + right) / 2
				if cumSumDist[mid] < threshold {
					left = mid + 1
				} else {
					selectedIdx = mid
					right = mid - 1
				}
			}

			candidateCentroid := make([][]float64, 1)
			candidateCentroid[0] = data[selectedIdx*km.numFeatures : (selectedIdx+1)*km.numFeatures]

			candidateDistances := make([]float64, N)
			for n := range N {
				candidateDistances[n] = distances[n]
			}
			candidateCumSumDist := make([]float64, N)

			loss := 0.0
			for n := range N {
				x := data[n*km.numFeatures : (n+1)*km.numFeatures]
				err := naiveMinIndeciesEarlyReturn(x, candidateCentroid, distances[n], func(minCol int, minDist float64) error {
					if minDist < distances[n] {
						loss += minDist
						candidateDistances[n] = minDist
					} else {
						loss += distances[n]
					}
					if n == 0 {
						candidateCumSumDist[n] = candidateDistances[n]
					} else {
						candidateCumSumDist[n] = candidateCumSumDist[n-1] + candidateDistances[n]
					}
					return nil
				})
				if err != nil {
					return
				}
			}
			if loss < bestLoss {
				bestLoss = loss
				bestSelectedIdx = selectedIdx
				bestCumSumDist = candidateCumSumDist
				bestDistances = candidateDistances
			}
		}
		distances = bestDistances
		cumSumDist = bestCumSumDist

		x := data[bestSelectedIdx*km.numFeatures : (bestSelectedIdx+1)*km.numFeatures]
		centroids = append(centroids, x)
		latestCentroid[0] = x
	}
	km.centroids = centroids
}

func naiveMinIndecies(x []float64, centroids [][]float64, fn func(minCol int, minDist float64) error) error {
	numClusters := len(centroids)
	minIdx := 0
	minVal := naiveSquaredEuclideanDistance(x, centroids[0])
	for k := 1; k < numClusters; k++ {
		val := naiveSquaredEuclideanDistanceEarlyReturn(x, centroids[k], minVal)
		if val < minVal {
			minVal = val
			minIdx = k
		}
	}
	return fn(minIdx, minVal)
}

func naiveMinIndeciesEarlyReturn(x []float64, centroids [][]float64, firstMinVal float64, fn func(minCol int, minDist float64) error) error {
	numClusters := len(centroids)
	minIdx := 0
	minVal := naiveSquaredEuclideanDistanceEarlyReturn(x, centroids[0], firstMinVal)
	for k := 1; k < numClusters; k++ {
		val := naiveSquaredEuclideanDistanceEarlyReturn(x, centroids[k], minVal)
		if val < minVal {
			minVal = val
			minIdx = k
		}
	}
	return fn(minIdx, minVal)
}

func naiveSquaredEuclideanDistance(x, y []float64) float64 {
	sum := 0.0
	for i := range len(x) {
		diff := x[i] - y[i]
		sum += diff * diff
	}
	return sum
}

func naiveSquaredEuclideanDistanceEarlyReturn(x, y []float64, minVal float64) float64 {
	sum := 0.0
	for i := range len(x) {
		diff := x[i] - y[i]
		sum += diff * diff
		if sum > minVal {
			return sum
		}
	}
	return sum
}
