// Copyright 2026 Skystarry-AI
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const net = std.net;
const posix = std.posix;
const HmacSha256 = std.crypto.auth.hmac.sha2.HmacSha256;

const PORT = 8080;
const BRAIN_SOCKET_PATH = "/tmp/tantalum-brain.sock";


// Reference:
// https://github.com/tupleapp/tuple-launch/blob/master/cmsghdr.zig
/// CMSG structure helper (compatible with 0.15.2)
pub fn Cmsghdr(comptime T: type) type {
    const Header = extern struct {
        len: usize,
        level: c_int,
        @"type": c_int,
    };

    const data_align = @sizeOf(usize);
    const data_offset = std.mem.alignForward(usize, @sizeOf(Header), data_align);

    return extern struct {
        const Self = @This();
        bytes: [data_offset + @sizeOf(T)]u8 align(@alignOf(Header)),

        pub fn init(args: struct { level: c_int, @"type": c_int, data: T }) Self {
            var self: Self = undefined;
            @memset(&self.bytes, 0);
            self.headerPtr().* = .{
                .len = data_offset + @sizeOf(T),
                .level = args.level,
                .@"type" = args.@"type",
            };
            self.dataPtr().* = args.data;
            return self;
        }

        pub fn headerPtr(self: *Self) *Header { return @ptrCast(self); }
        pub fn dataPtr(self: *Self) *align(data_align) T {
            return @ptrCast(@alignCast(&self.bytes[data_offset]));
        }
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Retrieve the one-time session key injected by Python from environment variables at startup
    const shared_secret = std.process.getEnvVarOwned(allocator, "TANTALUM_SECRET") catch |err| {
        std.debug.print("Fatal: TANTALUM_SECRET environment variable not provided: {any}\n", .{err});
        return err;
    };
    // Intentionally keep the memory allocated (do not defer free) as it needs to be referenced by threads throughout the program's lifetime.

    const address = try net.Address.parseIp4("127.0.0.1", PORT);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("Tantalum Gateway listening on 127.0.0.1:{d}\n", .{address.getPort()});

    while (true) {
        const connection = try server.accept();
        // Pass shared_secret along when spawning the thread
        const thread = try std.Thread.spawn(.{}, handleConnection, .{ allocator, connection, shared_secret });
        thread.detach();
    }
}

fn handleConnection(allocator: std.mem.Allocator, connection: net.Server.Connection, secret: []const u8) void {
    // Do not close the socket here because the Brain owns it after passing the FD.
    // Close only on verification failure or FD passing failure.

    var buffer: [4096]u8 = undefined;
    const bytes_read = connection.stream.read(&buffer) catch |err| {
        std.debug.print("Read error: {any}\n", .{err});
        connection.stream.close();
        return;
    };

    if (bytes_read < HmacSha256.mac_length) {
        connection.stream.close();
        return;
    }

    const received_mac = buffer[0..HmacSha256.mac_length];
    const payload = buffer[HmacSha256.mac_length..bytes_read];

    if (verifySpa(received_mac, payload, secret)) {
        std.debug.print("SPA Verified. Passing FD {}...\n", .{connection.stream.handle});

        passFdToBrain(allocator, connection.stream.handle) catch |err| {
            std.debug.print("Failed to pass FD: {any}\n", .{err});
            connection.stream.close();
        };
        // Do not close the socket upon successful FD passing - the Brain takes ownership
    } else {
        std.debug.print("SPA Verification Failed! Dropping connection.\n", .{});
        connection.stream.close();
    }
}

fn passFdToBrain(allocator: std.mem.Allocator, fd_to_pass: posix.fd_t) !void {
    const stream = try net.connectUnixSocket(BRAIN_SOCKET_PATH);
    defer stream.close();

    var msg_content = [1]u8{ 'F' };
    var iov = [_]posix.iovec_const{ .{ .base = &msg_content, .len = 1 } };

    const FdMessage = Cmsghdr(posix.fd_t);

    const cmsg_ptr = try allocator.create(FdMessage);
    defer allocator.destroy(cmsg_ptr);

    cmsg_ptr.* = FdMessage.init(.{
        .level = posix.SOL.SOCKET,
        .@"type" = 1, // SCM_RIGHTS (posix.SCM / linux.SCM.RIGHTS API not supported in 0.15.2)
        .data = fd_to_pass,
    });

    var msg = posix.msghdr_const{
        .name = null,
        .namelen = 0,
        .iov = &iov,
        .iovlen = iov.len,
        .control = @ptrCast(cmsg_ptr),
        .controllen = @intCast(@sizeOf(FdMessage)),
        .flags = 0,
    };

    _ = try posix.sendmsg(stream.handle, &msg, 0);
}

fn verifySpa(received_mac_slice: []const u8, payload: []const u8, secret: []const u8) bool {
    var expected_mac: [HmacSha256.mac_length]u8 = undefined;
    HmacSha256.create(&expected_mac, payload, secret);
    if (received_mac_slice.len != HmacSha256.mac_length) return false;
    // std.crypto.subtle / std.crypto.utils are missing in 0.15.2
    // Use std.crypto.timing_safe.eql
    return std.crypto.timing_safe.eql(
        [HmacSha256.mac_length]u8,
        expected_mac,
        received_mac_slice[0..HmacSha256.mac_length].*,
    );
}

test "SPA verification logic" {
    const secret = "test-dynamic-secret-key-12345678";
    const payload = "test-payload-data";
    var mac: [HmacSha256.mac_length]u8 = undefined;
    HmacSha256.create(&mac, payload, secret);

    // Test valid
    try std.testing.expect(verifySpa(&mac, payload, secret));

    // Test invalid payload
    try std.testing.expect(!verifySpa(&mac, "wrong-payload", secret));

    // Test invalid MAC
    var bad_mac = mac;
    bad_mac[0] ^= 0xFF;
    try std.testing.expect(!verifySpa(&bad_mac, payload, secret));
}
