/**
 * Streaming FNV-1a 64 over file bytes (matches ``sdx_native.native_tools.fnv1a64_file`` / Zig linecrc file mode).
 */
#ifndef SDX_FNV64_FILE_H
#define SDX_FNV64_FILE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(SDX_FNV64_FILE_BUILD)
#define SDX_FNV64_FILE_API __declspec(dllexport)
#elif defined(_WIN32)
#define SDX_FNV64_FILE_API __declspec(dllimport)
#else
#define SDX_FNV64_FILE_API
#endif

/**
 * @param out_hash FNV-1a 64 result
 * @param out_bytes total bytes read
 * @param out_newlines count of '\\n' bytes
 * @return 0 on success, -1 on open/read error
 */
SDX_FNV64_FILE_API int sdx_fnv1a64_file_stream(const char *utf8_path, uint64_t *out_hash, uint64_t *out_bytes,
                                               uint64_t *out_newlines);

#ifdef __cplusplus
}
#endif

#endif
