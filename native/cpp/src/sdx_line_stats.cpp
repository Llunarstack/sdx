#include "sdx/line_stats.h"

#include <cstdio>

int sdx_count_file_bytes_newlines(const char *utf8_path, unsigned long long *out_bytes,
                                  unsigned long long *out_newlines) {
    if (!utf8_path || !out_bytes || !out_newlines) {
        return -1;
    }
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
    unsigned long long nb = 0;
    unsigned long long nl = 0;
    for (;;) {
        size_t n = std::fread(buf, 1, sizeof(buf), fp);
        if (n == 0) {
            break;
        }
        nb += static_cast<unsigned long long>(n);
        for (size_t i = 0; i < n; ++i) {
            if (buf[i] == static_cast<unsigned char>('\n')) {
                ++nl;
            }
        }
    }
    const int err = std::ferror(fp);
    std::fclose(fp);
    if (err) {
        return -1;
    }
    *out_bytes = nb;
    *out_newlines = nl;
    return 0;
}
