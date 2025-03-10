package kmeans

import (
	"math"
	"slices"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func TestLinearAlgebraKMeansCentroids(t *testing.T) {
	numClusters := 3
	numFeatures := 3
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		t.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	initCentroids := [][]float64{
		{4.0, 5.0, 6.0},
		{1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0},
	}
	km.centroids = mat.NewDense(numClusters, numFeatures, slices.Concat(initCentroids...))

	X := []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
		-1.0, -2.0, -3.0,
	}
	err = km.Train(X, 1, 0.01)
	if err != nil {
		t.Fatalf("Failed to train NaiveKMeans: %v", err)
	}

	expectedCentroids := [][]float64{
		{5.5, 6.5, 7.5},
		{-1.0, -2.0, -3.0},
		{1.0, 2.0, 3.0},
	}

	centroids := km.Centroids()
	if len(centroids) != 3 {
		t.Fatalf("Expected 3 centroids, got %d", len(centroids))
	}
	for i, centroid := range centroids {
		if !floats.Equal(centroid, expectedCentroids[i]) {
			t.Fatalf("Expected centroid %d to be %v, got %v", i, expectedCentroids[i], centroid)
		}
	}

	predictions := make([]int, len(centroids))
	km.Predict(slices.Concat(centroids...), func(row, minCol int, minVal float64) error {
		predictions[row] = minCol
		return nil
	})

	expectedPredictions := []int{0, 1, 2}
	for i, prediction := range predictions {
		if prediction != expectedPredictions[i] {
			t.Fatalf("Expected prediction %d to be %d, got %d", i, expectedPredictions[i], prediction)
		}
	}
}
func TestLinearAlgebraKMeansPredict(t *testing.T) {
	trainX, err := generate4ClusterDataset(2, 5000) // 2dim, 20,000 samples
	if err != nil {
		t.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	km, err := NewLinearAlgebraKMeans(4, 2, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		t.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	err = km.Train(trainX, 100, 0.01)
	if err != nil {
		t.Fatalf("Failed to train NaiveKMeans: %v", err)
	}

	centroids := km.Centroids()
	testX := []float64{
		-5.0, -5.0,
		-5.0, 5.0,
		5.0, -5.0,
		5.0, 5.0,
	}
	epsilon := 0.05
	predictions := make([]int, len(testX)/2)
	km.Predict(testX, func(row, minCol int, minVal float64) error {
		predictions[row] = minCol
		x := testX[row*2 : (row+1)*2]
		if math.Abs(x[0]-centroids[minCol][0]) > epsilon || math.Abs(x[1]-centroids[minCol][1]) > epsilon {
			t.Fatalf("Expected x to be %v, got %v", centroids[minCol], x)
		}
		return nil
	})
}
