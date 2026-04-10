#ifndef SDX_C_BUFFER_STATS_H
#define SDX_C_BUFFER_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/**
 * Count ``\\n`` bytes in ``buf[0..len-1]`` (streaming-friendly).
 */
size_t sdx_c_count_newlines_u8(const uint8_t *buf, size_t len);

/**
 * Sum of all bytes as uint64 (checksum-style; not cryptographic).
 */
uint64_t sdx_c_sum_bytes_u8(const uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif /* SDX_C_BUFFER_STATS_H */
