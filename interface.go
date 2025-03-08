package kmeans

type KMeans interface {
	Train(data []float64, iter int) error
	Predict(data []float64, fn func(row, minCol int) bool) error
	Centroids() [][]float64
}
