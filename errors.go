package kmeans

import "fmt"

var ErrInvalidInitMethod = fmt.Errorf("invalid init method")

var ErrInvalidDataLength = fmt.Errorf("data length must be divisible by the number of features")

var ErrInvalidNumClusters = fmt.Errorf("number of clusters must be greater than 0")

var ErrInvalidNumFeatures = fmt.Errorf("number of features must be greater than 0")

var ErrEmptyData = fmt.Errorf("data is empty")

var ErrFewerClustersThanData = fmt.Errorf("number of clusters must be less than the number of data points")

var ErrInvalidMaxIterations = fmt.Errorf("max iterations must be greater than 0")

var ErrInvalidTolerance = fmt.Errorf("tolerance must be greater than 0")

var ErrInvalidConcurrency = fmt.Errorf("concurrency must be greater than 0")
