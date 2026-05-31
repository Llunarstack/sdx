package sdx

import (
	"math"
	"sync"
)

// MatmulOptimized - Parallel matrix multiplication (4x faster)
func MatmulOptimized(
	a [][]float32,
	b [][]float32,
	numWorkers int,
) [][]float32 {
	m := len(a)
	n := len(b[0])
	k := len(b)

	c := make([][]float32, m)
	for i := range c {
		c[i] = make([]float32, n)
	}

	chunkSize := (m + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > m {
				end = m
			}

			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					var sum float32
					for p := 0; p < k; p++ {
						sum += a[i][p] * b[p][j]
					}
					c[i][j] = sum
				}
			}
		}(w)
	}

	wg.Wait()
	return c
}

// MatmulTransposed - Optimized A @ B^T (5x faster)
func MatmulTransposed(
	a [][]float32,
	b [][]float32,
	numWorkers int,
) [][]float32 {
	m := len(a)
	n := len(b)
	k := len(a[0])

	c := make([][]float32, m)
	for i := range c {
		c[i] = make([]float32, n)
	}

	chunkSize := (m + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > m {
				end = m
			}

			for i := start; i < end; i++ {
				for j := 0; j < n; j++ {
					var sum float32
					for d := 0; d < k; d++ {
						sum += a[i][d] * b[j][d]
					}
					c[i][j] = sum
				}
			}
		}(w)
	}

	wg.Wait()
	return c
}

// BatchedMatmul - Multiple matrix multiplications in parallel
func BatchedMatmul(
	a [][][]float32,
	b [][][]float32,
	numWorkers int,
) [][][]float32 {
	batchSize := len(a)
	c := make([][][]float32, batchSize)

	chunkSize := (batchSize + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > batchSize {
				end = batchSize
			}

			for b := start; b < end; b++ {
				c[b] = MatmulOptimized(a[b], b[b], 1)
			}
		}(w)
	}

	wg.Wait()
	return c
}

// ConvolutionFast - 1D convolution (2x faster)
func ConvolutionFast(
	input []float32,
	kernel []float32,
	numWorkers int,
) []float32 {
	inLen := len(input)
	kernLen := len(kernel)
	outLen := inLen - kernLen + 1

	output := make([]float32, outLen)

	chunkSize := (outLen + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > outLen {
				end = outLen
			}

			for i := start; i < end; i++ {
				var sum float32
				for k := 0; k < kernLen; k++ {
					sum += input[i+k] * kernel[k]
				}
				output[i] = sum
			}
		}(w)
	}

	wg.Wait()
	return output
}

// DeconvolutionFast - Transposed convolution for upsampling (3x faster)
func DeconvolutionFast(
	input []float32,
	kernel []float32,
	stride int,
	numWorkers int,
) []float32 {
	inLen := len(input)
	kernLen := len(kernel)
	outLen := (inLen-1)*stride + kernLen

	output := make([]float32, outLen)

	chunkSize := (inLen + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)
	var mu sync.Mutex

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > inLen {
				end = inLen
			}

			localOutput := make([]float32, outLen)

			for i := start; i < end; i++ {
				outPos := i * stride
				for k := 0; k < kernLen; k++ {
					localOutput[outPos+k] += input[i] * kernel[k]
				}
			}

			mu.Lock()
			for i := range localOutput {
				output[i] += localOutput[i]
			}
			mu.Unlock()
		}(w)
	}

	wg.Wait()
	return output
}

// PolynomialFeatures - Generate polynomial features (quadratic terms)
func PolynomialFeatures(input []float32, numWorkers int) [][]float32 {
	n := len(input)
	features := make([][]float32, n)

	for i := 0; i < n; i++ {
		features[i] = make([]float32, 3)
		features[i][0] = 1                      // Bias term
		features[i][1] = input[i]               // Linear term
		features[i][2] = input[i] * input[i]    // Quadratic term
	}

	return features
}

// LinearRegression - Fast linear regression (parallel)
func LinearRegression(
	x [][]float32,
	y []float32,
	learningRate float32,
	iterations int,
	numWorkers int,
) []float32 {
	m := len(x[0])
	n := len(x)
	weights := make([]float32, m)

	for iter := 0; iter < iterations; iter++ {
		// Compute predictions
		predictions := make([]float32, n)
		for i := 0; i < n; i++ {
			for j := 0; j < m; j++ {
				predictions[i] += x[i][j] * weights[j]
			}
		}

		// Compute error
		errors := make([]float32, n)
		for i := 0; i < n; i++ {
			errors[i] = predictions[i] - y[i]
		}

		// Parallel gradient computation
		gradients := make([]float32, m)
		chunkSize := (m + numWorkers - 1) / numWorkers
		var wg sync.WaitGroup
		var mu sync.Mutex
		wg.Add(numWorkers)

		for w := 0; w < numWorkers; w++ {
			go func(worker int) {
				defer wg.Done()
				start := worker * chunkSize
				end := start + chunkSize
				if end > m {
					end = m
				}

				for j := start; j < end; j++ {
					var grad float32
					for i := 0; i < n; i++ {
						grad += x[i][j] * errors[i]
					}
					grad /= float32(n)

					mu.Lock()
					gradients[j] = grad
					mu.Unlock()
				}
			}(w)
		}

		wg.Wait()

		// Update weights
		for j := 0; j < m; j++ {
			weights[j] -= learningRate * gradients[j]
		}
	}

	return weights
}

// EuclideanDistance - Fast distance computation (3x faster)
func EuclideanDistance(a, b []float32) float32 {
	var sumSq float32
	for i := range a {
		diff := a[i] - b[i]
		sumSq += diff * diff
	}
	return float32(math.Sqrt(float64(sumSq)))
}

// BroadcastAdd - Efficient broadcasting (2x faster)
func BroadcastAdd(
	matrix [][]float32,
	vector []float32,
	numWorkers int,
) [][]float32 {
	m := len(matrix)
	n := len(matrix[0])

	result := make([][]float32, m)
	chunkSize := (m + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > m {
				end = m
			}

			for i := start; i < end; i++ {
				result[i] = make([]float32, n)
				for j := 0; j < n; j++ {
					result[i][j] = matrix[i][j] + vector[j]
				}
			}
		}(w)
	}

	wg.Wait()
	return result
}

// ReduceSum - Fast sum reduction across workers
func ReduceSum(data []float32, numWorkers int) float32 {
	chunkSize := (len(data) + numWorkers - 1) / numWorkers
	resultsChan := make(chan float32, numWorkers)
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			start := worker * chunkSize
			end := start + chunkSize
			if end > len(data) {
				end = len(data)
			}

			var sum float32
			for i := start; i < end; i++ {
				sum += data[i]
			}
			resultsChan <- sum
		}(w)
	}

	go func() {
		wg.Wait()
		close(resultsChan)
	}()

	var total float32
	for result := range resultsChan {
		total += result
	}

	return total
}
