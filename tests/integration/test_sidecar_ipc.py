"""Control-plane and shared-memory LOB feed contract tests for the Tachyon sidecar.

The sidecar no longer accepts inference requests over ZeroMQ: market data flows through
the wait-free SPSC ring in ``tachyon_lob_shm`` and ZeroMQ serves control only (PING /
STATS / SHUTDOWN). These tests act as the Python producer will in production — writing
LOBStateSlot frames straight into shared memory with the same seqlock protocol — then
verify batched inference accounting, logit numerics against the reference linear model,
and clean shutdown. If the sidecar is not running, tests skip instead of failing.
"""

from __future__ import annotations

import ctypes
import os
import struct
import time
from collections.abc import Iterator

import numpy as np
import pytest
import zmq

ENDPOINT = os.environ.get("TACHYON_SIDECAR_ENDPOINT", "tcp://127.0.0.1:5566")
CONNECT_TIMEOUT_S = 30.0
CALL_TIMEOUT_MS = 10_000

RING_CAPACITY = 65_536
CELL_TABLE_OFFSET = 192
CELL_STRIDE = 64
SLOT_PAYLOAD_OFFSET = 8
SLOT_DTYPE = np.dtype(
    [
        ("timestamp_ns", "<u8"),
        ("sequence", "<u8"),
        ("lob_state", "<f4", (8,)),
        ("flags", "<u4"),
        ("_reserved", "<u4"),
    ]
)
assert SLOT_DTYPE.itemsize == 56, "LOBStateSlot layout drift"

SEGMENT_BYTES = CELL_TABLE_OFFSET + CELL_STRIDE * RING_CAPACITY


def _reference_weights() -> tuple[np.ndarray, np.ndarray]:
    weights = np.zeros((8, 4), dtype=np.float32)
    flat = weights.ravel()
    for idx in range(flat.size):
        flat[idx] = ((idx * 37) % 17 - 8) / 16.0
    bias = (0.25 * np.arange(4, dtype=np.float32)) - 0.5
    return weights, bias


def expected_logits(state_row: np.ndarray) -> np.ndarray:
    weights, bias = _reference_weights()
    return state_row @ weights + bias


class ShmRingWriter:
    """Python mirror of SpscRingBuffer<LOBStateSlot, 65536> producer semantics."""

    def __init__(self) -> None:
        self._buf = self._map_segment()
        self._view = np.frombuffer(self._buf, dtype=np.uint8)
        self._cell_stamps = np.zeros(RING_CAPACITY, dtype=np.uint64)
        self.tail = 0

    @staticmethod
    def _map_segment() -> memoryview:
        if os.name == "nt":
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            FILE_MAP_ALL_ACCESS = 0xF001F  # noqa: N806 - Win32 API constant spelling
            kernel32.OpenFileMappingA.restype = ctypes.c_void_p
            kernel32.OpenFileMappingA.argtypes = [
                ctypes.c_uint32,
                ctypes.c_bool,
                ctypes.c_char_p,
            ]
            kernel32.MapViewOfFile.restype = ctypes.c_void_p
            kernel32.MapViewOfFile.argtypes = [
                ctypes.c_void_p,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_uint32,
                ctypes.c_size_t,
            ]
            handle = kernel32.OpenFileMappingA(FILE_MAP_ALL_ACCESS, False, b"tachyon_lob_shm")
            if not handle:
                raise FileNotFoundError("tachyon_lob_shm not found (sidecar not running?)")
            address = kernel32.MapViewOfFile(handle, FILE_MAP_ALL_ACCESS, 0, 0, SEGMENT_BYTES)
            if not address:
                raise OSError("MapViewOfFile failed")
            return memoryview((ctypes.c_char * SEGMENT_BYTES).from_address(address))
        fd = os.open("/dev/shm/tachyon_lob_shm", os.O_RDWR)
        import mmap

        return memoryview(mmap.mmap(fd, SEGMENT_BYTES))

    def push(self, timestamp_ns: int, sequence: int, lob_state: np.ndarray, flags: int = 0) -> None:
        cell = self.tail & (RING_CAPACITY - 1)
        base = CELL_TABLE_OFFSET + cell * CELL_STRIDE
        stamp = int(self._cell_stamps[cell])

        struct.pack_into("<Q", self._view, base, stamp + 1)
        payload = struct.pack(
            "<QQ8fI4x",
            timestamp_ns,
            sequence,
            *[float(v) for v in lob_state],
            flags,
        )
        payload_view = np.frombuffer(payload, dtype=np.uint8)
        self._view[base + SLOT_PAYLOAD_OFFSET : base + SLOT_PAYLOAD_OFFSET + len(payload)] = (
            payload_view
        )
        struct.pack_into("<Q", self._view, base, stamp + 2)
        self._cell_stamps[cell] = stamp + 2

        struct.pack_into("<Q", self._view, 0, self.tail + 1)
        self.tail += 1


@pytest.fixture(scope="module")
def client() -> Iterator[zmq.Socket[bytes]]:
    with zmq.Context() as ctx:
        sock: zmq.Socket[bytes] = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt(zmq.REQ_RELAXED, 1)
        sock.setsockopt(zmq.REQ_CORRELATE, 1)
        sock.connect(ENDPOINT)
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        ready = False
        while time.monotonic() < deadline:
            try:
                sock.send(b"PING")
                if sock.recv() == b"PONG":
                    ready = True
                    break
            except zmq.Again:
                pass
        if not ready:
            sock.close()
            pytest.skip(f"tachyon_sidecar not reachable at {ENDPOINT}")
        sock.setsockopt(zmq.RCVTIMEO, CALL_TIMEOUT_MS)
        yield sock
        sock.close()


def _stats(client: zmq.Socket[bytes]) -> dict[str, str]:
    client.send(b"STATS")
    reply = client.recv().decode("ascii")
    assert reply.startswith("STATS|"), reply
    fields = {}
    for part in reply.split("|")[1:]:
        key, _, value = part.partition("=")
        fields[key] = value
    return fields


def _parse_logits(fields: dict[str, str]) -> np.ndarray:
    return np.array([float(v) for v in fields["last_logits"].split(",")], dtype=np.float32)


class TestControlPlane:
    def test_ping_pong(self, client: zmq.Socket[bytes]) -> None:
        client.send(b"PING")
        assert client.recv() == b"PONG"

    def test_stats_contract(self, client: zmq.Socket[bytes]) -> None:
        fields = _stats(client)
        for key in ("batches", "inferences", "dropped", "last_batch", "last_logits"):
            assert key in fields
        assert int(fields["batches"]) >= 0
        logits = _parse_logits(fields)
        assert logits.size == 4
        assert np.isfinite(logits).all()


class TestSharedMemoryFeed:
    def test_end_to_end_batched_inference(self, client: zmq.Socket[bytes]) -> None:
        try:
            ring = ShmRingWriter()
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"shared segment unavailable: {exc}")

        before = _stats(client)
        rng = np.random.default_rng(42)
        n_frames = 2048
        for seq in range(int(before["inferences"]) + 1, int(before["inferences"]) + 1 + n_frames):
            row = rng.standard_normal(8).astype(np.float32)
            ring.push(time.monotonic_ns(), seq, row)

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            after = _stats(client)
            if int(after["inferences"]) >= int(before["inferences"]) + n_frames:
                break
            time.sleep(0.01)
        else:
            pytest.fail(f"sidecar never consumed {n_frames} fed slots")

        assert int(after["dropped"]) == int(before["dropped"]), "consumer lagged behind producer"
        assert int(after["last_batch"]) <= 16, "batch exceeded MAX_BATCH_SIZE"

        probe_row = np.linspace(-2.0, 2.0, num=8, dtype=np.float32)
        probe_batches_before = int(after["batches"])
        pushed_through = int(after["inferences"])

        deadline = time.monotonic() + 15.0
        solo_batch = False
        snapshot = {}
        while time.monotonic() < deadline:
            time.sleep(0.005)
            snapshot = _stats(client)
            if (
                int(snapshot["batches"]) > probe_batches_before
                and int(snapshot["last_batch"]) == 1
            ):
                solo_batch = True
                break
            if int(snapshot["inferences"]) >= pushed_through:
                probe_row = probe_row * 0.999
                pushed_through += 1
                ring.push(time.monotonic_ns(), pushed_through, probe_row)
        else:
            pytest.fail("never observed a solo probe batch")
        assert solo_batch

        got = _parse_logits(snapshot)
        want = expected_logits(probe_row)
        np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)

    def test_shutdown(self, client: zmq.Socket[bytes]) -> None:
        client.send(b"SHUTDOWN")
        assert client.recv() == b"BYE"
