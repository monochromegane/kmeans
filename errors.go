package kmeans

import "errors"

var (
	ErrInvalidInitMethod     = errors.New("invalid init method")
	ErrInvalidDataLength     = errors.New("data length must be divisible by the number of features")
	ErrInvalidNumClusters    = errors.New("number of clusters must be greater than 0")
	ErrInvalidNumFeatures    = errors.New("number of features must be greater than 0")
	ErrEmptyData             = errors.New("data is empty")
	ErrFewerClustersThanData = errors.New("number of clusters must be less than the number of data points")
	ErrInvalidMaxIterations  = errors.New("max iterations must be greater than 0")
	ErrInvalidTolerance      = errors.New("tolerance must be greater than 0")
	ErrInvalidConcurrency    = errors.New("concurrency must be greater than 0")
)
