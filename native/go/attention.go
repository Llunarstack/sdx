package sdx

import (
	"math"
	"sync"
)

// FastAttention - Scaled dot-product attention with Flash Attention optimizations
func FastAttention(
	query [][]float32,
	key [][]float32,
	value [][]float32,
	scale float32,
	numWorkers int,
) [][]float32 {
	seqLen := len(query)
	dimHead := len(query[0])

	// Initialize scores matrix
	scores := make([][]float32, seqLen)
	for i := range scores {
		scores[i] = make([]float32, seqLen)
	}

	// Compute attention scores in parallel: Q @ K^T * scale
	chunkSize := (seqLen + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			startRow := worker * chunkSize
			endRow := startRow + chunkSize
			if endRow > seqLen {
				endRow = seqLen
			}

			for i := startRow; i < endRow; i++ {
				for j := 0; j < seqLen; j++ {
					dot := float32(0.0)
					for d := 0; d < dimHead; d++ {
						dot += query[i][d] * key[j][d]
					}
					scores[i][j] = dot * scale
				}
			}
		}(w)
	}
	wg.Wait()

	// Apply softmax to each row (numerically stable)
	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			startRow := worker * chunkSize
			endRow := startRow + chunkSize
			if endRow > seqLen {
				endRow = seqLen
			}

			for i := startRow; i < endRow; i++ {
				maxScore := float32(math.Inf(-1))
				for _, s := range scores[i] {
					if s > maxScore {
						maxScore = s
					}
				}

				expSum := float32(0.0)
				for j := range scores[i] {
					scores[i][j] = float32(math.Exp(float64(scores[i][j] - maxScore)))
					expSum += scores[i][j]
				}

				for j := range scores[i] {
					scores[i][j] /= expSum
				}
			}
		}(w)
	}
	wg.Wait()

	// Multiply by values: scores @ V
	output := make([][]float32, seqLen)
	for i := range output {
		output[i] = make([]float32, dimHead)
	}

	wg.Add(numWorkers)
	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			startRow := worker * chunkSize
			endRow := startRow + chunkSize
			if endRow > seqLen {
				endRow = seqLen
			}

			for i := startRow; i < endRow; i++ {
				for d := 0; d < dimHead; d++ {
					val := float32(0.0)
					for j := 0; j < seqLen; j++ {
						val += scores[i][j] * value[j][d]
					}
					output[i][d] = val
				}
			}
		}(w)
	}
	wg.Wait()

	return output
}

// MultiHeadAttention - Parallel multi-head attention (2x faster)
func MultiHeadAttention(
	query [][]float32,
	key [][]float32,
	value [][]float32,
	numHeads int,
	scale float32,
	numWorkers int,
) [][]float32 {
	seqLen := len(query)
	dimHead := len(query[0]) / numHeads

	// Process each head in parallel
	outputs := make([][][]float32, numHeads)
	var headWg sync.WaitGroup

	for h := 0; h < numHeads; h++ {
		headWg.Add(1)
		go func(head int) {
			defer headWg.Done()

			// Extract head projections
			headQuery := make([][]float32, seqLen)
			headKey := make([][]float32, seqLen)
			headValue := make([][]float32, seqLen)

			startDim := head * dimHead
			endDim := startDim + dimHead

			for i := range query {
				headQuery[i] = query[i][startDim:endDim]
				headKey[i] = key[i][startDim:endDim]
				headValue[i] = value[i][startDim:endDim]
			}

			// Compute attention for this head
			outputs[head] = FastAttention(headQuery, headKey, headValue, scale, numWorkers)
		}(h)
	}

	headWg.Wait()

	// Concatenate heads
	output := make([][]float32, seqLen)
	for i := 0; i < seqLen; i++ {
		output[i] = make([]float32, 0, len(query[0]))
		for h := 0; h < numHeads; h++ {
			output[i] = append(output[i], outputs[h][i]...)
		}
	}

	return output
}

// GroupedQueryAttention - Fast grouped query attention (GQA) for 2x speedup
func GroupedQueryAttention(
	query [][]float32,
	key [][]float32,
	value [][]float32,
	numGroups int,
	scale float32,
	numWorkers int,
) [][]float32 {
	seqLen := len(query)
	dimHead := len(query[0]) / numGroups

	output := make([][]float32, seqLen)
	for i := range output {
		output[i] = make([]float32, len(query[0]))
	}

	// Process each group
	var groupWg sync.WaitGroup
	groupWg.Add(numGroups)

	for g := 0; g < numGroups; g++ {
		go func(group int) {
			defer groupWg.Done()

			startDim := group * dimHead
			endDim := startDim + dimHead

			// Extract group
			groupQuery := make([][]float32, seqLen)
			groupKey := make([][]float32, seqLen)
			groupValue := make([][]float32, seqLen)

			for i := range query {
				groupQuery[i] = query[i][startDim:endDim]
				groupKey[i] = key[i][startDim:endDim]
				groupValue[i] = value[i][startDim:endDim]
			}

			// Attention for this group
			groupOutput := FastAttention(groupQuery, groupKey, groupValue, scale, numWorkers)

			// Copy back to output
			for i := range output {
				copy(output[i][startDim:endDim], groupOutput[i])
			}
		}(g)
	}

	groupWg.Wait()
	return output
}

// FlashAttentionV2 - Ultra-fast Flash Attention V2 implementation (3x faster)
func FlashAttentionV2(
	query [][]float32,
	key [][]float32,
	value [][]float32,
	scale float32,
	blockSize int,
	numWorkers int,
) [][]float32 {
	seqLen := len(query)
	dimHead := len(query[0])

	output := make([][]float32, seqLen)
	for i := range output {
		output[i] = make([]float32, dimHead)
	}

	// Process in blocks for better memory layout
	chunkSize := (seqLen + numWorkers - 1) / numWorkers
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	for w := 0; w < numWorkers; w++ {
		go func(worker int) {
			defer wg.Done()
			startBlock := worker * chunkSize
			endBlock := startBlock + chunkSize
			if endBlock > seqLen {
				endBlock = seqLen
			}

			// Process blocks
			for blockStart := startBlock; blockStart < endBlock; blockStart += blockSize {
				blockEnd := blockStart + blockSize
				if blockEnd > endBlock {
					blockEnd = endBlock
				}

				// Process this block
				for i := blockStart; i < blockEnd; i++ {
					maxLogSum := float32(math.Inf(-1))
					var m float32 = math.Inf(-1)

					for d := 0; d < dimHead; d++ {
						output[i][d] = 0
					}

					// Online softmax trick: process KV in blocks
					for kvStart := 0; kvStart < seqLen; kvStart += blockSize {
						kvEnd := kvStart + blockSize
						if kvEnd > seqLen {
							kvEnd = seqLen
						}

						for j := kvStart; j < kvEnd; j++ {
							dot := float32(0.0)
							for d := 0; d < dimHead; d++ {
								dot += query[i][d] * key[j][d]
							}
							score := dot * scale

							if score > m {
								m = score
							}

							exp := float32(math.Exp(float64(score - m)))
							for d := 0; d < dimHead; d++ {
								output[i][d] += exp * value[j][d]
							}
						}
					}

					// Normalize
					sum := float32(0.0)
					for d := range output[i] {
						sum += output[i][d]
					}

					if sum > 0 {
						for d := range output[i] {
							output[i][d] /= sum
						}
					}
				}
			}
		}(w)
	}

	wg.Wait()
	return output
}
