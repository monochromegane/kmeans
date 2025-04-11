package kmeans

type NewOption func(*KMeansState) error

func WithInitMethod(method int) NewOption {
	return func(state *KMeansState) error {
		if method != INIT_NONE && method != INIT_KMEANS_PLUS_PLUS && method != INIT_RANDOM {
			return ErrInvalidInitMethod
		}
		state.InitMethod = method
		return nil
	}
}

type TrainOption func(*trainConfig) error

func WithMaxIterations(iter int) TrainOption {
	return func(config *trainConfig) error {
		if iter <= 0 {
			return ErrInvalidMaxIterations
		}
		config.iter = iter
		return nil
	}
}

func WithTolerance(tol float32) TrainOption {
	return func(config *trainConfig) error {
		if tol <= 0 {
			return ErrInvalidTolerance
		}
		config.tol = tol
		return nil
	}
}

func WithConcurrency(concurrency int) TrainOption {
	return func(config *trainConfig) error {
		if concurrency <= 0 {
			return ErrInvalidConcurrency
		}
		config.concurrency = concurrency
		return nil
	}
}
