package kmeans

import "fmt"

var ErrInvalidInitMethod = fmt.Errorf("invalid init method")

var ErrInvalidDataLength = fmt.Errorf("data length must be divisible by the number of features")

var ErrInvalidNumClusters = fmt.Errorf("number of clusters must be greater than 0")

var ErrInvalidNumFeatures = fmt.Errorf("number of features must be greater than 0")

var ErrEmptyData = fmt.Errorf("data is empty")
