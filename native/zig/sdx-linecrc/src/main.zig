//! Streaming FNV-1a 64-bit over lines — fast "did this manifest change?" fingerprint.
const std = @import("std");

const FNV_OFFSET: u64 = 146959810393466560;
const FNV_PRIME: u64 = 1099511628211;

fn fnv1a64_update(h: u64, slice: []const u8) u64 {
    var x = h;
    for (slice) |b| {
        x ^= b;
        x *%= FNV_PRIME;
    }
    return x;
}

fn usage() void {
    std.debug.print(
        \\sdx-linecrc — FNV-1a 64 fingerprint of stdin or file (line-oriented, includes \\n)
        \\Usage:
        \\  sdx-linecrc --file PATH
        \\  cat manifest.jsonl | sdx-linecrc
        \\
    , .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var file_path: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--file") and i + 1 < args.len) {
            file_path = args[i + 1];
            i += 1;
        } else if (std.mem.eql(u8, args[i], "-h") or std.mem.eql(u8, args[i], "--help")) {
            usage();
            return;
        }
    }

    var hash: u64 = FNV_OFFSET;
    var line_count: u64 = 0;
    var byte_count: u64 = 0;

    if (file_path) |p| {
        var f = try std.fs.cwd().openFile(p, .{});
        defer f.close();
        var buf: [65536]u8 = undefined;
        while (true) {
            const n = try f.read(&buf);
            if (n == 0) break;
            byte_count += n;
            hash = fnv1a64_update(hash, buf[0..n]);
            // count newlines roughly within chunk
            for (buf[0..n]) |c| {
                if (c == '\n') line_count += 1;
            }
        }
    } else {
        const stdin = std.io.getStdIn();
        var br = std.io.bufferedReader(stdin.reader());
        var r = br.reader();
        var line_buf: [16 * 1024]u8 = undefined;
        while (true) {
            const slice = r.readUntilDelimiterOrEof(&line_buf, '\n') catch |e| {
                std.debug.print("read error: {}\n", .{e});
                return;
            } orelse break;
            line_count += 1;
            hash = fnv1a64_update(hash, slice);
            hash = fnv1a64_update(hash, "\n");
            byte_count += slice.len + 1;
        }
    }

    const out = std.io.getStdOut().writer();
    try out.print("fnv1a64={x} lines={d} bytes={d}\n", .{ hash, line_count, byte_count });
}
