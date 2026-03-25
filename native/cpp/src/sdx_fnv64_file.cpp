#include "sdx/fnv64_file.h"

#include <cstdio>

namespace {
constexpr uint64_t kFnvOffset = 146959810393466560ULL;
constexpr uint64_t kFnvPrime = 1099511628211ULL;
} // namespace

int sdx_fnv1a64_file_stream(const char *utf8_path, uint64_t *out_hash, uint64_t *out_bytes, uint64_t *out_newlines) {
    if (!utf8_path || !out_hash || !out_bytes || !out_newlines) {
        return -1;
    }
    *out_hash = kFnvOffset;
    *out_bytes = 0;
    *out_newlines = 0;

#if defined(_WIN32)
    FILE *fp = nullptr;
    if (fopen_s(&fp, utf8_path, "rb") != 0 || !fp) {
        return -1;
    }
#else
    FILE *fp = std::fopen(utf8_path, "rb");
    if (!fp) {
        return -1;
    }
#endif

    unsigned char buf[1 << 20];
    uint64_t h = kFnvOffset;
    uint64_t nb = 0;
    uint64_t nl = 0;
    for (;;) {
        size_t n = std::fread(buf, 1, sizeof(buf), fp);
        if (n == 0) {
            break;
        }
        nb += static_cast<uint64_t>(n);
        for (size_t i = 0; i < n; ++i) {
            const unsigned char b = buf[i];
            h ^= static_cast<uint64_t>(b);
            h = (h * kFnvPrime) & 0xFFFFFFFFFFFFFFFFULL;
            if (b == static_cast<unsigned char>('\n')) {
                ++nl;
            }
        }
    }
    const int err = std::ferror(fp);
    std::fclose(fp);
    if (err) {
        return -1;
    }
    *out_hash = h;
    *out_bytes = nb;
    *out_newlines = nl;
    return 0;
}
