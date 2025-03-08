package kmeans

type KMeans interface {
	Train(data []float64, iter int, tol float64) error
	Predict(data []float64, fn func(row, minCol int) error) error
	Centroids() [][]float64
}
