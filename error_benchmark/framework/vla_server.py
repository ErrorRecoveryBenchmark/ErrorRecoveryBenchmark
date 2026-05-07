#!/usr/bin/env python
"""
VLA Policy Server - VLA policy inference server

A standalone TCP server that runs in the VLA conda env (openpi / openpi05),
loads a model checkpoint, and serves action predictions over a socket.

Protocol (pickle over TCP, Python stdlib only):
    Request → dict, Response → dict
    ─────────────────────────────────────────────────────
    {"cmd": "predict", "obs": dict, "prompt": str}
        → {"action": np.ndarray}          # (action_dim,) or (chunk, action_dim)
    {"cmd": "reset"}
        → {"status": "ok"}
    {"cmd": "info"}
        → {"model_type": str, "action_dim": int, "chunk_size": int, ...}
    {"cmd": "shutdown"}
        → server exits
    ─────────────────────────────────────────────────────

Supported model types:
    pi0      – Pi0 (LIBERO fine-tune)
    pi05     – Pi0.5 base / fine-tune
    phoenix  – Phoenix / Flare variants (8-dim action)

Usage (from the VLA conda env):
    python -m error_benchmark.framework.vla_server \\
        --model_type pi0 \\
        --checkpoint ~/.cache/openpi/openpi-assets/checkpoints/pi0_libero/ \\
        --port 5555 --device cuda:0
"""

import argparse
import logging
import pickle
import queue
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════
# Wire protocol helpers
# ═══════════════════════════════════════════════════════

def _send(sock: socket.socket, obj: Any) -> None:
    """Send a pickle-serialized object prefixed with a 4-byte length header."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)


def _recv(sock: socket.socket) -> Any:
    """Receive a length-prefixed pickle object."""
    raw_len = _recvall(sock, 4)
    if raw_len is None:
        return None
    msg_len = struct.unpack("!I", raw_len)[0]
    data = _recvall(sock, msg_len)
    if data is None:
        return None
    return pickle.loads(data)


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """Receive exactly *n* bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


# ═══════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════

class BaseModelWrapper:
    """Interface every model loader must implement."""

    def __init__(self, checkpoint: str, device: str = "cuda:0"):
        self.checkpoint = checkpoint
        self.device = device
        self.action_dim: int = 7
        self.chunk_size: int = 50
        self.model_type: str = "base"

    def load(self) -> None:
        raise NotImplementedError

    def predict(self, obs: dict, prompt: str) -> np.ndarray:
        """Return (chunk_size, action_dim) action chunk."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset any episode-level state (e.g. action queues)."""
        pass

    def info(self) -> dict:
        return {
            "model_type": self.model_type,
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "checkpoint": self.checkpoint,
            "device": self.device,
        }


class OpenPiModelWrapper(BaseModelWrapper):
    """
    Wrapper for OpenPI Pi0 / Pi0.5 / Phoenix models.

    Uses ``create_trained_policy()`` from openpi which handles:
    - Model config & architecture
    - Checkpoint loading
    - Normalization (norm_stats) automatically
    - Input/output transforms

    Expects the ``openpi05`` conda environment.
    """

    def __init__(self, checkpoint: str, model_type: str = "pi0",
                 config_name: str = "pi0_libero", device: str = "cuda:0"):
        super().__init__(checkpoint, device)
        self.model_type = model_type
        self.config_name = config_name
        self._model = None

        # Pi0/Pi0.5 = 7-dim, Phoenix/Flare = 8-dim
        if "phoenix" in model_type or "flare" in model_type:
            self.action_dim = 8
        else:
            self.action_dim = 7

    def load(self) -> None:
        ckpt_dir = Path(self.checkpoint)
        logger.info(f"Loading {self.model_type} (config={self.config_name}) from {ckpt_dir}")

        # Attempt to import openpi inference utilities
        try:
            from openpi.training.config import get_config
            from openpi.policies.policy_config import create_trained_policy
            logger.info("openpi imports succeeded")
        except ImportError:
            logger.warning(
                "openpi not importable — falling back to dummy model. "
                "Make sure you are running in the openpi05 conda env."
            )
            self._model = None
            return

        # create_trained_policy handles norm_stats, transforms, etc. automatically
        try:
            config = get_config(self.config_name)
            self._model = create_trained_policy(
                train_config=config,
                checkpoint_dir=str(ckpt_dir),
            )
            logger.info(f"Model loaded successfully (config={self.config_name})")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self._model = None

    # Camera name → openpi key mapping per config
    # pi0_libero expects: observation/image (base), observation/wrist_image (wrist)
    # Default image key mapping (agentview → observation/image, wrist → observation/wrist_image)
    _DEFAULT_IMAGE_KEY_MAP = {
        "agentview": "observation/image",
        "robot0_eye_in_hand": "observation/wrist_image",
    }
    IMAGE_KEY_MAP = {
        "pi0_libero": _DEFAULT_IMAGE_KEY_MAP,
        "pi05_base": _DEFAULT_IMAGE_KEY_MAP,
        "pi05_libero": _DEFAULT_IMAGE_KEY_MAP,
    }

    def _build_example(self, obs: dict, prompt: str) -> dict:
        """Build the observation dict expected by openpi's infer().

        Args:
            obs: {"state": np.ndarray (8,), "images": {"agentview": (H,W,3), ...}}
            prompt: task prompt string

        Returns:
            dict suitable for ``self._model.infer(example)``
        """
        example = {"prompt": prompt}

        # Attach state
        if "state" in obs:
            example["observation/state"] = np.asarray(obs["state"], dtype=np.float32)

        # Attach images with correct key mapping for this config
        images = obs.get("images", {})
        key_map = self.IMAGE_KEY_MAP.get(self.config_name, self._DEFAULT_IMAGE_KEY_MAP)
        for cam_name, img in images.items():
            openpi_key = key_map.get(cam_name, f"observation/image/{cam_name}")
            example[openpi_key] = np.asarray(img, dtype=np.uint8)

        return example

    def predict(self, obs: dict, prompt: str) -> np.ndarray:
        """
        Run model inference.

        Args:
            obs: {"state": np.ndarray (8,), "images": {"agentview": (H,W,3), ...}}
            prompt: task prompt string

        Returns:
            (chunk_size, action_dim) action chunk
        """
        if self._model is None:
            logger.warning("Model not loaded, returning zero actions")
            return np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)

        example = self._build_example(obs, prompt)

        # Inference — create_trained_policy handles normalization internally
        result = self._model.infer(example)
        actions = np.asarray(result["actions"], dtype=np.float32)

        return actions

    def predict_batch(self, obs_list: List[dict], prompts: List[str]) -> List[np.ndarray]:
        """
        Run batched model inference on multiple observations.

        Applies input transforms per-item, stacks into a batch for a single
        forward pass, then splits and applies output transforms per-item.

        Args:
            obs_list: list of N obs dicts
            prompts: list of N prompt strings

        Returns:
            list of N action arrays, each (chunk_size, action_dim)
        """
        n = len(obs_list)
        if n == 0:
            return []

        if self._model is None:
            logger.warning("Model not loaded, returning zero actions for batch")
            return [np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
                    for _ in range(n)]

        # For single items, just use regular predict
        if n == 1:
            return [self.predict(obs_list[0], prompts[0])]

        # Build per-item examples and apply input transforms individually
        # (transforms may mutate in-place, so must be per-item)
        examples = [self._build_example(obs, prompt)
                    for obs, prompt in zip(obs_list, prompts)]

        # Try batched inference via infer_batch if available
        try:
            results = self._model.infer_batch(examples)
            return [np.asarray(r["actions"], dtype=np.float32) for r in results]
        except AttributeError:
            pass

        # Fallback: sequential inference (still benefits from request batching
        # to reduce network round-trips)
        logger.debug("infer_batch not available, falling back to sequential")
        actions_list = []
        for example in examples:
            result = self._model.infer(example)
            actions_list.append(np.asarray(result["actions"], dtype=np.float32))
        return actions_list

    def reset(self) -> None:
        pass

    def info(self) -> dict:
        base = super().info()
        base["config_name"] = self.config_name
        return base


MODEL_REGISTRY: Dict[str, type] = {
    "pi0": OpenPiModelWrapper,
    "pi05": OpenPiModelWrapper,
    "phoenix": OpenPiModelWrapper,
    "flare": OpenPiModelWrapper,
}

# Default config_name for each model_type (can be overridden via --config_name)
DEFAULT_CONFIG_NAMES: Dict[str, str] = {
    "pi0": "pi0_libero",
    "pi05": "pi05_libero",
    "phoenix": "pi0_libero",   # phoenix uses pi0 config with different checkpoint
    "flare": "pi0_libero",
}


# ═══════════════════════════════════════════════════════
# TCP Server
# ═══════════════════════════════════════════════════════

class VLAServer:
    """Single-client TCP server that wraps a model and serves predictions."""

    def __init__(self, model: BaseModelWrapper, host: str = "0.0.0.0",
                 port: int = 5555):
        self.model = model
        self.host = host
        self.port = port
        self._running = False

    def serve_forever(self) -> None:
        self._running = True
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        logger.info(f"VLA server listening on {self.host}:{self.port}")
        logger.info(f"Model: {self.model.info()}")

        while self._running:
            srv.settimeout(1.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            logger.info(f"Client connected: {addr}")
            self._handle_client(conn)
            conn.close()
            logger.info(f"Client disconnected: {addr}")

        srv.close()
        logger.info("Server shut down")

    def _handle_client(self, conn: socket.socket) -> None:
        """Process commands from a single client until disconnect or shutdown."""
        while self._running:
            try:
                request = _recv(conn)
            except (ConnectionError, EOFError, OSError):
                break

            if request is None:
                break

            cmd = request.get("cmd", "")
            try:
                if cmd == "predict":
                    obs = request.get("obs", {})
                    prompt = request.get("prompt", "")
                    t0 = time.time()
                    actions = self.model.predict(obs, prompt)
                    dt = time.time() - t0
                    logger.debug(f"predict: {dt:.3f}s, shape={actions.shape}")
                    _send(conn, {"action": actions})

                elif cmd == "reset":
                    self.model.reset()
                    _send(conn, {"status": "ok"})

                elif cmd == "info":
                    _send(conn, self.model.info())

                elif cmd == "shutdown":
                    _send(conn, {"status": "shutting_down"})
                    self._running = False
                    break

                else:
                    _send(conn, {"error": f"Unknown command: {cmd}"})

            except Exception as e:
                logger.error(f"Error handling '{cmd}': {e}", exc_info=True)
                try:
                    _send(conn, {"error": str(e)})
                except (ConnectionError, EOFError, OSError):
                    break


# ═══════════════════════════════════════════════════════
# Batched TCP Server (multi-client, batched inference)
# ═══════════════════════════════════════════════════════

@dataclass
class InferenceRequest:
    """A pending inference request from a client handler thread."""
    client_id: int
    obs: dict
    prompt: str
    result_event: threading.Event = field(default_factory=threading.Event)
    result: Optional[np.ndarray] = None
    error: Optional[str] = None


class BatchedVLAServer:
    """Multi-client TCP server with batched GPU inference.

    Architecture:
        - Accept thread: listens for new connections, spawns handler threads
        - Client handler threads (N): recv request → put InferenceRequest on queue
          → wait for result_event → send response
        - Batch inference thread: collects requests from queue → batched forward pass
          → distributes results → signals events

    Thread safety: queue.Queue is thread-safe. result_event is per-request.
    The model is only accessed from the batch inference thread.
    """

    def __init__(self, model: BaseModelWrapper, host: str = "0.0.0.0",
                 port: int = 5555, max_clients: int = 16,
                 batch_timeout_ms: int = 20, max_batch_size: int = 16):
        self.model = model
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self._running = False
        self._request_queue: queue.Queue = queue.Queue()
        self._client_counter = 0
        self._client_lock = threading.Lock()
        self._active_clients = 0
        self._stats_lock = threading.Lock()
        self._total_predictions = 0
        self._total_batches = 0

    def serve_forever(self) -> None:
        self._running = True

        # Start batch inference thread
        batch_thread = threading.Thread(
            target=self._batch_inference_loop, name="batch-inference", daemon=True
        )
        batch_thread.start()

        # Start accept loop
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(self.max_clients)
        logger.info(
            f"Batched VLA server listening on {self.host}:{self.port} "
            f"(max_clients={self.max_clients}, max_batch={self.max_batch_size}, "
            f"batch_timeout={self.batch_timeout_ms}ms)"
        )
        logger.info(f"Model: {self.model.info()}")

        while self._running:
            srv.settimeout(1.0)
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue

            with self._client_lock:
                self._client_counter += 1
                client_id = self._client_counter

            logger.info(f"Client {client_id} connected: {addr}")

            handler = threading.Thread(
                target=self._handle_client,
                args=(conn, addr, client_id),
                name=f"client-{client_id}",
                daemon=True,
            )
            handler.start()

        srv.close()
        logger.info("Batched server shut down")

    def _handle_client(self, conn: socket.socket, addr: tuple, client_id: int) -> None:
        """Process commands from a single client (runs in its own thread)."""
        with self._stats_lock:
            self._active_clients += 1

        try:
            while self._running:
                try:
                    request = _recv(conn)
                except (ConnectionError, EOFError, OSError):
                    break

                if request is None:
                    break

                cmd = request.get("cmd", "")
                try:
                    if cmd == "predict":
                        obs = request.get("obs", {})
                        prompt = request.get("prompt", "")

                        # Submit to batch queue and wait
                        req = InferenceRequest(
                            client_id=client_id,
                            obs=obs,
                            prompt=prompt,
                        )
                        self._request_queue.put(req)

                        # Wait for batch thread to fill in result (timeout 60s)
                        if not req.result_event.wait(timeout=60.0):
                            _send(conn, {"error": "Inference timeout (60s)"})
                            continue

                        if req.error is not None:
                            _send(conn, {"error": req.error})
                        else:
                            _send(conn, {"action": req.result})

                    elif cmd == "reset":
                        # Reset is per-client; model.reset() is a no-op for OpenPI
                        _send(conn, {"status": "ok"})

                    elif cmd == "info":
                        info = self.model.info()
                        info["batched"] = True
                        info["max_clients"] = self.max_clients
                        info["max_batch_size"] = self.max_batch_size
                        with self._stats_lock:
                            info["active_clients"] = self._active_clients
                            info["total_predictions"] = self._total_predictions
                            info["total_batches"] = self._total_batches
                        _send(conn, info)

                    elif cmd == "shutdown":
                        _send(conn, {"status": "shutting_down"})
                        self._running = False
                        break

                    else:
                        _send(conn, {"error": f"Unknown command: {cmd}"})

                except Exception as e:
                    logger.error(f"Client {client_id} error on '{cmd}': {e}", exc_info=True)
                    try:
                        _send(conn, {"error": str(e)})
                    except (ConnectionError, EOFError, OSError):
                        break
        finally:
            conn.close()
            with self._stats_lock:
                self._active_clients -= 1
            logger.info(f"Client {client_id} disconnected: {addr}")

    def _batch_inference_loop(self) -> None:
        """Collects requests from queue and runs batched inference."""
        timeout_s = self.batch_timeout_ms / 1000.0

        while self._running:
            batch: List[InferenceRequest] = []

            # Wait for first request (blocking with 1s timeout for shutdown check)
            try:
                first = self._request_queue.get(timeout=1.0)
                batch.append(first)
            except queue.Empty:
                continue

            # Drain additional requests up to max_batch_size or batch_timeout
            deadline = time.monotonic() + timeout_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    req = self._request_queue.get(timeout=remaining)
                    batch.append(req)
                except queue.Empty:
                    break

            # Run batched inference
            batch_size = len(batch)
            t0 = time.time()
            try:
                obs_list = [r.obs for r in batch]
                prompts = [r.prompt for r in batch]
                results = self.model.predict_batch(obs_list, prompts)

                for req, actions in zip(batch, results):
                    req.result = actions
                    req.result_event.set()

            except Exception as e:
                logger.error(f"Batch inference error: {e}", exc_info=True)
                for req in batch:
                    req.error = str(e)
                    req.result_event.set()

            dt = time.time() - t0
            with self._stats_lock:
                self._total_predictions += batch_size
                self._total_batches += 1

            logger.debug(
                f"Batch: size={batch_size}, time={dt:.3f}s, "
                f"per_item={dt/batch_size:.3f}s"
            )


# ═══════════════════════════════════════════════════════
# Client helper (used by PolicyServerAdapter)
# ═══════════════════════════════════════════════════════

class VLAClient:
    """Thin TCP client for communicating with a VLAServer."""

    def __init__(self, host: str = "localhost", port: int = 5555,
                 timeout: float = 30.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self._sock: Optional[socket.socket] = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        logger.info(f"Connected to VLA server at {self.host}:{self.port}")

    def close(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None

    def _ensure_connected(self) -> None:
        if self._sock is None:
            self.connect()

    def predict(self, obs: dict, prompt: str) -> np.ndarray:
        """Send obs to server, receive action chunk."""
        self._ensure_connected()
        _send(self._sock, {"cmd": "predict", "obs": obs, "prompt": prompt})
        resp = _recv(self._sock)
        if resp is None:
            raise ConnectionError("Server disconnected")
        if "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp["action"]

    def reset(self) -> None:
        self._ensure_connected()
        _send(self._sock, {"cmd": "reset"})
        resp = _recv(self._sock)
        if resp and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")

    def info(self) -> dict:
        self._ensure_connected()
        _send(self._sock, {"cmd": "info"})
        resp = _recv(self._sock)
        if resp is None:
            raise ConnectionError("Server disconnected")
        return resp

    def shutdown(self) -> None:
        self._ensure_connected()
        _send(self._sock, {"cmd": "shutdown"})
        try:
            _recv(self._sock)
        except Exception:
            pass
        self.close()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()


# ═══════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="VLA Policy Server")
    parser.add_argument("--model_type", type=str, default="pi0",
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model type to load")
    parser.add_argument("--config_name", type=str, default=None,
                        help="OpenPI config name (e.g. pi0_libero, pi05_base). "
                             "If not specified, uses default for model_type.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint directory")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Bind address")
    parser.add_argument("--port", type=int, default=5555,
                        help="TCP port")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for model inference")
    # Batched server options
    parser.add_argument("--batched", action="store_true",
                        help="Enable batched multi-client mode for parallel evaluation")
    parser.add_argument("--max_clients", type=int, default=16,
                        help="Max concurrent client connections (batched mode)")
    parser.add_argument("--max_batch_size", type=int, default=8,
                        help="Max batch size for inference (batched mode)")
    parser.add_argument("--batch_timeout_ms", type=int, default=20,
                        help="Max wait time (ms) to collect a batch (batched mode)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Resolve checkpoint path
    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    # Resolve config_name
    config_name = args.config_name or DEFAULT_CONFIG_NAMES.get(args.model_type, "pi0_libero")

    # Create and load model
    wrapper_cls = MODEL_REGISTRY[args.model_type]
    model = wrapper_cls(
        checkpoint=str(ckpt),
        model_type=args.model_type,
        config_name=config_name,
        device=args.device,
    )
    model.load()

    # Start server (batched or single-client)
    if args.batched:
        server = BatchedVLAServer(
            model,
            host=args.host,
            port=args.port,
            max_clients=args.max_clients,
            max_batch_size=args.max_batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
        )
    else:
        server = VLAServer(model, host=args.host, port=args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted, shutting down")


if __name__ == "__main__":
    main()
