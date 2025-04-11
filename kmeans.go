package kmeans

import (
	"encoding/gob"
	"math"
	"math/rand/v2"
	"runtime"

	"golang.org/x/sync/errgroup"
)

type KMeans struct {
	state *KMeansState
}

type KMeansState struct {
	InitMethod  int
	NumClusters int
	NumFeatures int
	Centroids   [][]float32
}

func NewKMeans(numClusters, numFeatures, initMethod int) (*KMeans, error) {
	if numClusters <= 0 {
		return nil, ErrInvalidNumClusters
	}
	if numFeatures <= 0 {
		return nil, ErrInvalidNumFeatures
	}
	if initMethod != INIT_NONE && initMethod != INIT_KMEANS_PLUS_PLUS && initMethod != INIT_RANDOM {
		return nil, ErrInvalidInitMethod
	}

	return &KMeans{
		state: &KMeansState{
			InitMethod:  initMethod,
			NumClusters: numClusters,
			NumFeatures: numFeatures,
			Centroids:   make([][]float32, numClusters),
		},
	}, nil
}

func LoadKMeans(dec *gob.Decoder) (*KMeans, error) {
	state := &KMeansState{}
	err := dec.Decode(state)
	if err != nil {
		return nil, err
	}
	return &KMeans{
		state: state,
	}, nil
}

func (km *KMeans) Train(data []float32, iter int, tol float32) (int, float32, error) {
	if len(data) == 0 {
		return 0, 0.0, ErrEmptyData
	}
	if len(data)%km.state.NumFeatures != 0 {
		return 0, 0.0, ErrInvalidDataLength
	}
	if km.state.NumClusters > len(data)/km.state.NumFeatures {
		return 0, 0.0, ErrFewerClustersThanData
	}

	if km.state.InitMethod == INIT_RANDOM {
		km.initializeRandom(data)
	} else if km.state.InitMethod == INIT_KMEANS_PLUS_PLUS {
		km.initializeGreedyKMeansPlusPlus(data)
	}

	loss := float32(math.Inf(1))
	numIter := 0
	N := int(len(data) / km.state.NumFeatures)

	numWorkers := runtime.NumCPU()
	chunkSize := N / numWorkers
	if chunkSize == 0 {
		chunkSize = 1
		numWorkers = N
	}

	type result struct {
		centroids [][]float32
		counts    []int
		loss      float32
	}

	var eg errgroup.Group
	eg.SetLimit(numWorkers)

	for i := range iter {
		results := make(chan result, numWorkers)
		for w := range numWorkers {
			centroids := km.state.Centroids
			start := w * chunkSize
			end := start + chunkSize
			if w == numWorkers-1 {
				end = N
			}
			eg.Go(func() error {
				counts := make([]int, km.state.NumClusters)
				newCentroids := make([][]float32, km.state.NumClusters)
				for k := range km.state.NumClusters {
					newCentroids[k] = make([]float32, km.state.NumFeatures)
				}
				loss := float32(0.0)
				for n := start; n < end; n++ {
					x := data[n*km.state.NumFeatures : (n+1)*km.state.NumFeatures]

					err := minIndecies(x, centroids, func(minCol int, minDist float32) error {
						for d := range km.state.NumFeatures {
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

		counts := make([]int, km.state.NumClusters)
		newCentroids := make([][]float32, km.state.NumClusters)
		for k := range km.state.NumClusters {
			newCentroids[k] = make([]float32, km.state.NumFeatures)
		}
		newLoss := float32(0.0)
		for r := range results {
			for k := range km.state.NumClusters {
				for d := range km.state.NumFeatures {
					newCentroids[k][d] += r.centroids[k][d]
				}
				counts[k] += r.counts[k]
			}
			newLoss += r.loss
		}
		loss = newLoss

		frobNorm := float32(0.0)
		centroidDiff := float32(0.0)
		for k := range km.state.NumClusters {
			if counts[k] == 0 {
				continue
			}
			for d := range km.state.NumFeatures {
				newCentroids[k][d] /= float32(counts[k])

				newCentroid := newCentroids[k][d]
				diff := km.state.Centroids[k][d] - newCentroid
				centroidDiff += diff * diff
				frobNorm += newCentroid * newCentroid
			}
		}

		km.state.Centroids = newCentroids
		numIter = i

		if math.Sqrt(float64(centroidDiff))/(math.Sqrt(float64(frobNorm))) < float64(tol) {
			break
		}
	}

	return numIter, loss, nil
}

func (km *KMeans) Predict(data []float32, fn func(row, minCol int, minVal float32) error) error {
	if len(data) == 0 {
		return ErrEmptyData
	}
	if len(data)%km.state.NumFeatures != 0 {
		return ErrInvalidDataLength
	}

	N := int(len(data) / km.state.NumFeatures)
	for n := range N {
		x := data[n*km.state.NumFeatures : (n+1)*km.state.NumFeatures]

		err := minIndecies(x, km.state.Centroids, func(minCol int, minDist float32) error {
			return fn(n, minCol, minDist)
		})
		if err != nil {
			return err
		}
	}
	return nil
}

func (km *KMeans) Centroids() [][]float32 {
	viewCentroids := make([][]float32, len(km.state.Centroids))
	for i, centroid := range km.state.Centroids {
		viewCentroids[i] = make([]float32, len(centroid))
		copy(viewCentroids[i], centroid)
	}
	return viewCentroids
}

func (km *KMeans) Encode(enc *gob.Encoder) error {
	return enc.Encode(km.state)
}

func (km *KMeans) Decode(dec *gob.Decoder) error {
	return dec.Decode(km.state)
}

func (km *KMeans) initializeRandom(data []float32) {
	N := int(len(data) / km.state.NumFeatures)
	indecies := rand.Perm(N)[:km.state.NumClusters]
	for i, idx := range indecies {
		km.state.Centroids[i] = data[idx*km.state.NumFeatures : (idx+1)*km.state.NumFeatures]
	}
}

func (km *KMeans) initializeGreedyKMeansPlusPlus(data []float32) {
	L := min(2+int(math.Log(float64(km.state.NumClusters))), km.state.NumClusters, km.state.NumFeatures)

	N := int(len(data) / km.state.NumFeatures)
	idx := rand.IntN(N)
	distances := make([]float32, N)
	centroids := [][]float32{
		data[idx*km.state.NumFeatures : (idx+1)*km.state.NumFeatures],
	}
	latestCentroid := make([][]float32, 1)
	latestCentroid[0] = centroids[0]

	cumSumDist := make([]float32, N)
	for n := range N {
		distances[n] = float32(math.Inf(1))
		x := data[n*km.state.NumFeatures : (n+1)*km.state.NumFeatures]
		minIndeciesEarlyReturn(x, latestCentroid, distances[n], func(minCol int, minDist float32) error {
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

	var bestCumSumDist []float32
	var bestDistances []float32
	for k := 1; k < km.state.NumClusters; k++ {
		bestLoss := float32(math.Inf(1))
		bestSelectedIdx := 0
		for range L {
			threshold := rand.Float32() * cumSumDist[N-1]

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

			candidateCentroid := make([][]float32, 1)
			candidateCentroid[0] = data[selectedIdx*km.state.NumFeatures : (selectedIdx+1)*km.state.NumFeatures]

			candidateDistances := make([]float32, N)
			for n := range N {
				candidateDistances[n] = distances[n]
			}
			candidateCumSumDist := make([]float32, N)

			loss := float32(0.0)
			for n := range N {
				x := data[n*km.state.NumFeatures : (n+1)*km.state.NumFeatures]
				err := minIndeciesEarlyReturn(x, candidateCentroid, distances[n], func(minCol int, minDist float32) error {
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

		x := data[bestSelectedIdx*km.state.NumFeatures : (bestSelectedIdx+1)*km.state.NumFeatures]
		centroids = append(centroids, x)
		latestCentroid[0] = x
	}
	km.state.Centroids = centroids
}

func minIndecies(x []float32, centroids [][]float32, fn func(minCol int, minDist float32) error) error {
	numClusters := len(centroids)
	minIdx := 0
	minVal := squaredEuclideanDistance(x, centroids[0])
	for k := 1; k < numClusters; k++ {
		val := squaredEuclideanDistanceEarlyReturn(x, centroids[k], minVal)
		if val < minVal {
			minVal = val
			minIdx = k
		}
	}
	return fn(minIdx, minVal)
}

func minIndeciesEarlyReturn(x []float32, centroids [][]float32, firstMinVal float32, fn func(minCol int, minDist float32) error) error {
	numClusters := len(centroids)
	minIdx := 0
	minVal := squaredEuclideanDistanceEarlyReturn(x, centroids[0], firstMinVal)
	for k := 1; k < numClusters; k++ {
		val := squaredEuclideanDistanceEarlyReturn(x, centroids[k], minVal)
		if val < minVal {
			minVal = val
			minIdx = k
		}
	}
	return fn(minIdx, minVal)
}

func squaredEuclideanDistance(x, y []float32) float32 {
	sum := float32(0.0)
	for i := range len(x) {
		diff := x[i] - y[i]
		sum += diff * diff
	}
	return sum
}

func squaredEuclideanDistanceEarlyReturn(x, y []float32, minVal float32) float32 {
	sum := float32(0.0)
	for i := range len(x) {
		diff := x[i] - y[i]
		sum += diff * diff
		if sum > minVal {
			return sum
		}
	}
	return sum
}
