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
