package kmeans

import (
	"testing"
)

func TestNaiveKMeansInterface(t *testing.T) {
	var km KMeans
	km, _ = NewNaiveKMeans(3, 2, INIT_KMEANS_PLUS_PLUS)
	if _, ok := km.(*NaiveKMeans); !ok {
		t.Fatalf("km is not a NaiveKMeans")
	}
}
func TestLinearAlgebraKMeansInterface(t *testing.T) {
	var km KMeans
	km, _ = NewLinearAlgebraKMeans(3, 2, INIT_KMEANS_PLUS_PLUS)
	if _, ok := km.(*LinearAlgebraKMeans); !ok {
		t.Fatalf("km is not a LinearAlgebraKMeans")
	}
}
