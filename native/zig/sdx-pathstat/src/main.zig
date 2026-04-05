//! One path per line → TAB columns: path, size_bytes, status (ok|missing|error)
//! Resolves relative paths against the current working directory.
const std = @import("std");

fn usage() void {
    std.debug.print(
        \\sdx-pathstat — stat files listed one path per line
        \\  sdx-pathstat --file paths.txt
        \\  type paths.txt | sdx-pathstat
        \\
    , .{});
}

fn trimLine(buf: []const u8) []const u8 {
    var start: usize = 0;
    var end: usize = buf.len;
    while (start < end and (buf[start] == ' ' or buf[start] == '\t' or buf[start] == '\r')) {
        start += 1;
    }
    while (end > start and (buf[end - 1] == ' ' or buf[end - 1] == '\t' or buf[end - 1] == '\r')) {
        end -= 1;
    }
    return buf[start..end];
}

fn statAndPrint(writer: anytype, path: []const u8) !void {
    if (path.len == 0) return;
    const file = std.fs.cwd().openFile(path, .{}) catch {
        try writer.print("{s}\t0\tmissing\n", .{path});
        return;
    };
    defer file.close();
    const st = try file.stat();
    try writer.print("{s}\t{d}\tok\n", .{ path, st.size });
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

    const out = std.io.getStdOut().writer();

    if (file_path) |p| {
        var f = try std.fs.cwd().openFile(p, .{});
        defer f.close();
        var buf: [4096 * 4]u8 = undefined;
        var br = std.io.bufferedReader(f.reader());
        var r = br.reader();
        while (true) {
            const slice = r.readUntilDelimiterOrEof(&buf, '\n') catch |e| {
                std.debug.print("read error: {}\n", .{e});
                return;
            } orelse break;
            const line = trimLine(slice);
            if (line.len == 0) continue;
            statAndPrint(out, line) catch |e| {
                try out.print("{s}\t0\terror:{}\n", .{ line, @errorName(e) });
            };
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
            const line = trimLine(slice);
            if (line.len == 0) continue;
            statAndPrint(out, line) catch |e| {
                try out.print("{s}\t0\terror:{}\n", .{ line, @errorName(e) });
            };
        }
    }
}
