/**
 * Fast manifest / text file stats (bytes + newline count, wc -l compatible).
 * C ABI for Python ctypes — useful for huge JSONL without parsing JSON.
 */
#ifndef SDX_LINE_STATS_H
#define SDX_LINE_STATS_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_LINE_STATS_BUILD)
#define SDX_LINE_STATS_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_LINE_STATS_API __declspec(dllimport)
#else
#define SDX_LINE_STATS_API
#endif

/**
 * Count total bytes read and newline (0x0A) bytes in a file.
 * Compatible with ``wc -l`` when every record line ends with ``\\n``.
 *
 * @return 0 on success, -1 if file cannot be opened/read
 */
SDX_LINE_STATS_API int sdx_count_file_bytes_newlines(const char *utf8_path, unsigned long long *out_bytes,
                                                     unsigned long long *out_newlines);

#ifdef __cplusplus
}
#endif

#endif /* SDX_LINE_STATS_H */
