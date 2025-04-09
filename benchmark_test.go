package kmeans

import (
	"math/rand/v2"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func BenchmarkNaiveKMeansClusters4Datapoints10000Features2InitRandom(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 0
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features2InitRandom(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 0
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features2InitKMeansPlusPlus(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 0
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features2InitKMeansPlusPlus(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 0
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features2Iter1(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 1
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	km.state.centroids = make([][]float64, numClusters)
	for k := 0; k < numClusters; k++ {
		km.state.centroids[k] = make([]float64, numFeatures)
		for d := 0; d < numFeatures; d++ {
			km.state.centroids[k][d] = rand.Float64()
		}
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features2Iter1(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 1
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	centroids := mat.NewDense(numClusters, numFeatures, nil)
	for k := 0; k < numClusters; k++ {
		for d := 0; d < numFeatures; d++ {
			centroids.Set(k, d, rand.Float64())
		}
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features2InitRandomTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 1000
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features2InitRandomTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 1000
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features2InitKMeansPlusPlusTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 1000
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features2InitKMeansPlusPlusTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 2
	numDatapoints := 10000
	numIter := 1000
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features1024InitRandom(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 0
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features1024InitRandom(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 0
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features1024InitKMeansPlusPlus(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 0
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features1024InitKMeansPlusPlus(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 0
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features1024Iter1(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 1
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	km.state.centroids = make([][]float64, numClusters)
	for k := 0; k < numClusters; k++ {
		km.state.centroids[k] = make([]float64, numFeatures)
		for d := 0; d < numFeatures; d++ {
			km.state.centroids[k][d] = rand.Float64()
		}
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features1024Iter1(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 1
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_NONE)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}
	centroids := mat.NewDense(numClusters, numFeatures, nil)
	for k := 0; k < numClusters; k++ {
		for d := 0; d < numFeatures; d++ {
			centroids.Set(k, d, rand.Float64())
		}
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, -1.0) // -1.0 means no tolerance
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features1024InitRandomTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 1000
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features1024InitRandomTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 1000
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_RANDOM)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkNaiveKMeansClusters4Datapoints10000Features1024InitKMeansPlusPlusTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 1000
	km, err := NewNaiveKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}

func BenchmarkLinearAlgebraKMeansClusters4Datapoints10000Features1024InitKMeansPlusPlusTol1e6(b *testing.B) {
	numClusters := 4
	numFeatures := 1024
	numDatapoints := 10000
	numIter := 1000
	km, err := NewLinearAlgebraKMeans(numClusters, numFeatures, INIT_KMEANS_PLUS_PLUS)
	if err != nil {
		b.Fatalf("Failed to create NaiveKMeans: %v", err)
	}

	data, err := generate4ClusterDataset(numFeatures, numDatapoints/numClusters)
	if err != nil {
		b.Fatalf("Failed to generate 4 cluster dataset: %v", err)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		km.Train(data, numIter, 1e-6)
	}
}
