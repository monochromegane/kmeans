package kmeans

import "encoding/gob"

type KMeans interface {
	Train(data []float64, iter int, tol float64) (int, float64, error)
	Predict(data []float64, fn func(row, minCol int, minVal float64) error) error
	Centroids() [][]float64
	Encode(enc *gob.Encoder) error
	Decode(dec *gob.Decoder) error
}
