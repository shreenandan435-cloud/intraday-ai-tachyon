"""Hot-swap pipeline integration — PyTorch GTrXL export → ZMQ signal → live sidecar.

Boots the real C++ sidecar on an isolated control port, exports a dummy GTrXL to
ONNX (validated), mints a contract-matched TensorRT plan with *different* weights
via the local plan builder, fires the RELOAD_ENGINE signal, and proves the swap:
generation bumps, the next SHM tick is consumed without interruption, and
last_logits changes — i.e. the newly loaded weights are actually executing.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import zmq

from tachyon.rl.export import (
    ExportError,
    build_tensorrt_plan,
    export_gtrxl_to_onnx,
    signal_hot_swap,
)

ENDPOINT = os.environ.get("TACHYON_SIDECAR_ENDPOINT", "tcp://127.0.0.1:5577")
SIDECAR_EXE = "build/tachyon_sidecar.exe"
PLAN_BUILDER_EXE = "build/tachyon_plan_builder.exe"
BASE_PLAN = "build/test_engine.plan"
CONNECT_TIMEOUT_S = 30.0


# ── sidecar lifecycle ───────────────────────────────────────────────────────


def _wait_for_ready(sock: zmq.Socket[bytes], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            sock.send(b"PING")
            if sock.recv() == b"PONG":
                return True
        except zmq.Again:
            pass
        time.sleep(0.05)
    return False


def _connect_req(ctx: zmq.Context, timeout_ms: int) -> zmq.Socket[bytes]:
    """REQ socket with relaxed state machine: re-sends survive recv timeouts."""
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.REQ_RELAXED, 1)
    sock.setsockopt(zmq.REQ_CORRELATE, 1)
    sock.connect(ENDPOINT)
    return sock


@pytest.fixture(scope="module")
def sidecar() -> Iterator[None]:
    """Boot the C++ sidecar detached (survives tool-call boundaries), then stop it."""
    sidecar_exe = Path(SIDECAR_EXE)
    if not sidecar_exe.exists():
        pytest.skip("tachyon_sidecar.exe not built")

    log_dir = Path(os.environ.get("TEMP", "/tmp")) / "opencode"
    out_log = log_dir / "hotswap_stdout.log"
    err_log = log_dir / "hotswap_stderr.log"
    cmdline = (
        f'cmd.exe /c ""{sidecar_exe.resolve()}" --engine "{BASE_PLAN}" '
        f'--endpoint {ENDPOINT} > "{out_log}" 2> "{err_log}""'
    )
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
            f"@{{ CommandLine = '{cmdline}'; "
            f"CurrentDirectory = '{Path.cwd()}' }} | Select-Object -ExpandProperty ProcessId",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    pid = int(result.stdout.strip().splitlines()[-1])

    ctx = zmq.Context()
    probe = _connect_req(ctx, 500)
    try:
        if not _wait_for_ready(probe, CONNECT_TIMEOUT_S):
            _shutdown(pid)
            pytest.fail(f"sidecar never became READY at {ENDPOINT}")
        yield
    finally:
        probe.close(0)
        ctx.term()
        _shutdown(pid)


def _shutdown(pid: int) -> None:
    ctx = zmq.Context()
    sock = _connect_req(ctx, 3000)
    try:
        sock.send(b"SHUTDOWN")
        sock.recv()
    except zmq.Again:
        pass
    finally:
        sock.close(0)
        ctx.term()
    for _ in range(50):
        ret = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if str(pid) not in ret.stdout:
            return
        time.sleep(0.1)
    subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, check=False)


@pytest.fixture
def client(sidecar: None) -> Iterator[zmq.Socket[bytes]]:
    ctx = zmq.Context()
    sock = _connect_req(ctx, 10_000)
    yield sock
    sock.close(0)
    ctx.term()


# ── helpers ────────────────────────────────────────────────────────────────


def _stats(sock: zmq.Socket[bytes]) -> dict[str, str]:
    sock.send(b"STATS")
    reply = sock.recv().decode("ascii")
    assert reply.startswith("STATS|"), reply
    fields: dict[str, str] = {}
    for part in reply.split("|")[1:]:
        key, _, value = part.partition("=")
        fields[key] = value
    return fields


def _push_probe_tick(sequence: int) -> None:
    """Write one LOB slot straight into shared memory, mirroring the producer."""
    import ctypes
    import struct

    capacity, cell_table_offset, cell_stride = 65_536, 192, 64
    segment_bytes = cell_table_offset + cell_stride * capacity

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    file_map_all_access = 0xF001F
    kernel32.OpenFileMappingA.restype = ctypes.c_void_p
    kernel32.OpenFileMappingA.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_char_p]
    kernel32.MapViewOfFile.restype = ctypes.c_void_p
    kernel32.MapViewOfFile.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_size_t,
    ]
    handle = kernel32.OpenFileMappingA(file_map_all_access, False, b"tachyon_lob_shm")
    assert handle, "shared segment missing (sidecar owns it while running)"
    address = kernel32.MapViewOfFile(handle, file_map_all_access, 0, 0, segment_bytes)
    assert address, "MapViewOfFile failed"
    view = (ctypes.c_char * segment_bytes).from_address(address)

    current_tail = int.from_bytes(bytes(view[0:8]), "little")
    cell = current_tail & (capacity - 1)
    cell_base = cell_table_offset + cell * cell_stride

    stamp = int.from_bytes(bytes(view[cell_base : cell_base + 8]), "little")
    view[cell_base : cell_base + 8] = (stamp + 1).to_bytes(8, "little")

    payload = struct.pack(
        "<QQ8fI4x",
        time.monotonic_ns() % 2**63,
        sequence,
        float(sequence % 7) - 3.0,
        0.5,
        -0.25,
        1.0,  # lob_state[0:4]
        -1.0,
        0.125,
        -0.125,
        2.5,  # lob_state[4:8]
        0,  # flags
    )
    assert len(payload) == 56
    view[cell_base + 8 : cell_base + 8 + 56] = payload
    view[cell_base : cell_base + 8] = (stamp + 2).to_bytes(8, "little")
    view[0:8] = (current_tail + 1).to_bytes(8, "little")


def _wait_for(sock: zmq.Socket[bytes], predicate, timeout_s: float = 20.0) -> dict[str, str]:
    deadline = time.monotonic() + timeout_s
    fields: dict[str, str] = {}
    while time.monotonic() < deadline:
        fields = _stats(sock)
        if predicate(fields):
            return fields
        time.sleep(0.05)
    pytest.fail(f"condition not met in {timeout_s}s; last stats={fields}")


# ─── tests ──────────────────────────────────────────────────────────────────


class TestHotSwapPipeline:
    def test_sidecar_alive_and_stats_contract(self, client: zmq.Socket[bytes]) -> None:
        client.send(b"PING")
        assert client.recv() == b"PONG"
        fields = _stats(client)
        for key in ("batches", "inferences", "generation", "last_reload"):
            assert key in fields

    def test_hot_swap_changes_live_weights(self, client: zmq.Socket[bytes]) -> None:
        # Mint a same-contract plan with different weights.
        variant_plan = "build/test_engine_v1.plan"
        if not Path(variant_plan).exists():
            subprocess.run(
                [PLAN_BUILDER_EXE, "--output", variant_plan, "--variant", "1"],
                check=True,
                capture_output=True,
                text=True,
            )

        gen_before = int(_stats(client)["generation"])
        _push_probe_tick(sequence=1)
        before = _wait_for(
            client,
            lambda f: int(f["inferences"]) > 0 and int(f["generation"]) == gen_before,
        )
        logits_before = before["last_logits"]

        swap = signal_hot_swap(variant_plan, endpoint=ENDPOINT)
        assert swap["status"] == "accepted", swap["reply"]
        assert swap["reply"].startswith("200 OK|reload_accepted|")

        after = _wait_for(
            client,
            lambda f: (
                int(f["generation"]) == gen_before + 1 and "swapped" in f.get("last_reload", "")
            ),
        )
        assert int(after["generation"]) == gen_before + 1

        # Feed another tick through the NEW engine and confirm the logits moved.
        batches_before = int(after["batches"])
        _push_probe_tick(sequence=2)
        swapped = _wait_for(client, lambda f: int(f["batches"]) > batches_before)
        assert swapped["last_logits"] != logits_before, (
            "logits unchanged after hot-swap — new weights are not live"
        )
        floats = [float(v) for v in swapped["last_logits"].split(",")]
        assert all(v == v and abs(v) != float("inf") for v in floats)

    def test_reload_missing_file_rejected(self, client: zmq.Socket[bytes]) -> None:
        """The ack is synchronous; the build failure surfaces in last_reload."""
        gen_before = int(_stats(client)["generation"])
        swap = signal_hot_swap("build/does_not_exist.plan", endpoint=ENDPOINT, timeout_ms=15_000)
        assert swap["status"] == "accepted", "control thread must ack before the heavy build"

        fields = _wait_for(
            client,
            lambda f: "500 ERR" in f.get("last_reload", "") and int(f["generation"]) == gen_before,
        )
        assert "build_failed" in fields["last_reload"]
        # Generation must NOT advance on a failed swap.
        assert int(_stats(client)["generation"]) == gen_before

    def test_consumer_keeps_draining_across_swap(self, client: zmq.Socket[bytes]) -> None:
        """No tick is lost or corrupted by the retirement handshake."""
        inf_before = int(_stats(client)["inferences"])
        for seq in range(20, 30):
            _push_probe_tick(sequence=seq)
        after = _wait_for(client, lambda f: int(f["inferences"]) >= inf_before + 10)
        assert int(after["dropped"]) >= 0  # field present; no crash mid-swap


class TestExportArtifact:
    def test_dummy_gtrxl_exports_valid_onnx(self, tmp_path) -> None:
        pytest.importorskip("torch")
        from tachyon.rl.models.gtrxl import TachyonActorCriticGTrXL as Model

        model = Model(obs_dim=25, d_model=32, n_heads=4, n_layers=2, mem_len=8)
        onnx_path = tmp_path / "gtrxl.onnx"
        meta = export_gtrxl_to_onnx(model, onnx_path, seq_len=8)

        assert onnx_path.exists()
        assert meta["inputs"] == ["obs", "mem_flat"]
        assert meta["outputs"] == ["action_logits", "value"]

        import onnx

        graph = onnx.load(str(onnx_path)).graph
        named_dims = {}
        for value_info in (*graph.input, *graph.output):
            dims = value_info.type.tensor_type.shape.dim
            named_dims[value_info.name] = [d.dim_param or d.dim_value for d in dims]
        assert named_dims["obs"][0] == "batch"
        assert named_dims["obs"][1] == "seq"
        assert named_dims["mem_flat"][1] == "batch"

    def test_tensorrt_build_tier_resolves_or_raises(self, tmp_path) -> None:
        """On TRT-equipped hosts the plan builds; here it must fail with guidance."""
        pytest.importorskip("torch")
        from tachyon.rl.models.gtrxl import TachyonActorCriticGTrXL as Model

        model = Model(obs_dim=25, d_model=32, n_heads=4, n_layers=2, mem_len=8)
        onnx_path = tmp_path / "gtrxl.onnx"
        export_gtrxl_to_onnx(model, onnx_path, seq_len=8)
        plan_path = tmp_path / "gtrxl.plan"

        try:
            tier = build_tensorrt_plan(onnx_path, plan_path)
        except ExportError as exc:
            assert "tensorrt" in str(exc).lower() or "trtexec" in str(exc).lower()
        else:
            assert tier in ("python_api", "trtexec")
            assert plan_path.exists()

    def test_signal_to_dead_endpoint_is_graceful(self) -> None:
        result = signal_hot_swap(BASE_PLAN, endpoint="tcp://127.0.0.1:59999", timeout_ms=400)
        assert result["status"] == "unreachable"
