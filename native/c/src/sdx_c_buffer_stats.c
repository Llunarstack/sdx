/*
 * Build (example): cc -O3 -shared -fPIC -I../include -o libsdx_c_buffer_stats.so sdx_c_buffer_stats.c
 * Windows: cl /LD /O2 /I..\include sdx_c_buffer_stats.c
 * Install the resulting DLL/SO next to other sdx native artifacts under native/cpp/build/Release.
 */
#include "../include/sdx_c_buffer_stats.h"

size_t sdx_c_count_newlines_u8(const uint8_t *buf, size_t len) {
    size_t n = 0;
    if (!buf) {
        return 0;
    }
    for (size_t i = 0; i < len; ++i) {
        if (buf[i] == (uint8_t)'\n') {
            ++n;
        }
    }
    return n;
}

uint64_t sdx_c_sum_bytes_u8(const uint8_t *buf, size_t len) {
    uint64_t acc = 0;
    if (!buf) {
        return 0;
    }
    for (size_t i = 0; i < len; ++i) {
        acc += (uint64_t)buf[i];
    }
    return acc;
}
