package kmeans

import (
	"bytes"
	"encoding/gob"
	"math"
	"slices"
	"testing"
)

func TestNaiveKMeansCentroids(t *testing.T) {
	numClusters := 3
	numFeatures := 3
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		t.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	initCentroids := [][]float64{
		{4.0, 5.0, 6.0},
		{1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0},
	}
	km.state.Centroids = initCentroids

	X := []float64{
		1.0, 2.0, 3.0,
		4.0, 5.0, 6.0,
		7.0, 8.0, 9.0,
		-1.0, -2.0, -3.0,
	}
	_, _, err = km.Train(X, 1, 0.01)
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
		if !floatsEqual(centroid, expectedCentroids[i]) {
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

func TestNaiveKMeansEncodeDecode(t *testing.T) {
	numClusters := 3
	numFeatures := 3
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		t.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	initCentroids := [][]float64{
		{4.0, 5.0, 6.0},
		{1.0, 1.0, 1.0},
		{2.0, 2.0, 2.0},
	}
	km.state.Centroids = initCentroids

	buf := new(bytes.Buffer)
	err = km.Encode(gob.NewEncoder(buf))
	if err != nil {
		t.Fatalf("Failed to encode NaiveKMeans: %v", err)
	}

	km2, err := LoadNaiveKMeans(gob.NewDecoder(buf))
	if err != nil {
		t.Fatalf("Failed to load NaiveKMeans: %v", err)
	}

	if km.state.InitMethod != km2.state.InitMethod {
		t.Fatalf("Expected initMethod to be %d, got %d", km.state.InitMethod, km2.state.InitMethod)
	}
	if km.state.NumClusters != km2.state.NumClusters {
		t.Fatalf("Expected numClusters to be %d, got %d", km.state.NumClusters, km2.state.NumClusters)
	}
	if km.state.NumFeatures != km2.state.NumFeatures {
		t.Fatalf("Expected numFeatures to be %d, got %d", km.state.NumFeatures, km2.state.NumFeatures)
	}
	for c := range numClusters {
		if !floatsEqual(km.state.Centroids[c], km2.state.Centroids[c]) {
			t.Fatalf("Expected centroids to be %v, got %v", km.state.Centroids, km2.state.Centroids)
		}
	}
}

func floatsEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	const epsilon = 1e-14
	for i := range a {
		if math.Abs(a[i]-b[i]) > epsilon {
			return false
		}
	}
	return true
}
