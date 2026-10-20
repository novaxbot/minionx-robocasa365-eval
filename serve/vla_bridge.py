#!/usr/bin/env python3
"""VLA bridge: Xiaomi RoboCasa365 processor on CPU here, model on the GPU fleet
behind the loopback gateway (length-prefixed pickle protocol, one request in
flight per connection). HTTP API (ThreadingHTTPServer):
  GET  /healthz -> "ok"
  POST /act  body = pickle {"instruction": str,
                            "images": {key: uint8[T,H,W,3]} (3 camera keys),
                            "states": float32[T,14]}
             -> pickle float32[N,12] decoded policy actions (N = chunk length)
"""
import argparse
import collections
import io
import json
import pickle
import queue
import socket
import struct
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image
from transformers import AutoProcessor

CAMERA_KEYS = (
    "video.robot0_agentview_left",
    "video.robot0_agentview_right",
    "video.robot0_eye_in_hand",
)
STATE_DIM = 60
ACTION_DIM = 12
ROBOT_TYPE = "robocasa365"


def center_crop(image, crop_ratio):
    pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if crop_ratio >= 1.0:
        return pil_image
    width, height = pil_image.size
    crop_width = max(1, int(width * crop_ratio))
    crop_height = max(1, int(height * crop_ratio))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = pil_image.crop((left, top, left + crop_width, top + crop_height))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return cropped.resize((width, height), resampling)


class Conn:
    def __init__(self, host, port, timeout):
        self.host, self.port, self.timeout = host, port, timeout
        self.sock = None

    def _connect(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)

    def close(self):
        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass
        self.sock = None

    def _recv(self, n):
        buf = b""
        while len(buf) < n:
            packet = self.sock.recv(n - len(buf))
            if not packet:
                raise ConnectionError("backend closed the connection")
            buf += packet
        return buf

    def call(self, data):
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        for attempt in range(2):
            try:
                if self.sock is None:
                    self._connect()
                self.sock.sendall(struct.pack(">I", len(payload)) + payload)
                length = struct.unpack(">I", self._recv(4))[0]
                return pickle.loads(self._recv(length))
            except Exception:
                self.close()
                if attempt == 1:
                    raise


class Pool:
    def __init__(self, host, port, size, timeout):
        self.q = queue.Queue()
        for _ in range(size):
            self.q.put(Conn(host, port, timeout))

    def call(self, data):
        conn = self.q.get()
        try:
            return conn.call(data)
        finally:
            self.q.put(conn)


class Bridge:
    def __init__(self, model_path, gateway_host, gateway_port, conns, crop_ratio, timeout):
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if ROBOT_TYPE not in self.processor.list_robot_types():
            raise RuntimeError("robot type missing from checkpoint")
        self.pool = Pool(gateway_host, gateway_port, conns, timeout)
        self.crop_ratio = crop_ratio
        self.lat = collections.deque(maxlen=500)
        self.n = 0
        self.errors = 0
        self.lock = threading.Lock()

    def _messages(self, images, instruction):
        videos = {k: [center_crop(f, self.crop_ratio) for f in images[k]] for k in CAMERA_KEYS}
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Left camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[0]]},
                    {"type": "text", "text": "\nRight camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[1]]},
                    {"type": "text", "text": "\nWrist camera: "},
                    {"type": "video", "video": videos[CAMERA_KEYS[2]]},
                    {"type": "text", "text": f"\n\nGenerate robot actions for the task:\n{instruction} /no_cot"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "<cot></cot>"}]},
        ]

    def act(self, instruction, images, states):
        states = np.asarray(states, dtype=np.float32)
        state = np.zeros((1, states.shape[0], STATE_DIM), dtype=np.float32)
        state[0, :, : states.shape[-1]] = states
        inputs = self.processor.apply_chat_template(
            self._messages(images, instruction), tokenize=True, return_dict=True,
            return_tensors="pt", do_resize=False, state=state, robot_type=ROBOT_TYPE,
        )
        request = dict(inputs)
        request["task_id"] = ROBOT_TYPE
        raw = self.pool.call(request)
        actions = self.processor.decode_action(raw, robot_type=ROBOT_TYPE)
        return np.asarray(actions[0, :, :ACTION_DIM].float().cpu().numpy(), dtype=np.float32)


BRIDGE = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/octet-stream"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/healthz":
            return self._send(200, b"ok", "text/plain")
        if self.path == "/stats":
            lat = list(BRIDGE.lat)
            body = json.dumps({"requests": BRIDGE.n, "errors": BRIDGE.errors,
                               "latency_ms_mean": round(1000 * sum(lat) / len(lat), 1) if lat else None,
                               "latency_ms_p90": round(1000 * sorted(lat)[int(len(lat) * 0.9)], 1) if lat else None}).encode()
            return self._send(200, body, "application/json")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/act":
            return self._send(404, b"not found", "text/plain")
        t0 = time.time()
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = pickle.loads(self.rfile.read(n))
            images = req.get("images")
            if images is None and req.get("images_jpeg"):
                images = {k: np.stack([np.asarray(Image.open(io.BytesIO(b)).convert("RGB"), dtype=np.uint8) for b in frames], axis=0)
                          for k, frames in req["images_jpeg"].items()}
            actions = BRIDGE.act(str(req["instruction"]), images, req["states"])
            body = pickle.dumps(actions, protocol=pickle.HIGHEST_PROTOCOL)
            with BRIDGE.lock:
                BRIDGE.n += 1
                BRIDGE.lat.append(time.time() - t0)
            self._send(200, body)
        except Exception as exc:
            with BRIDGE.lock:
                BRIDGE.errors += 1
            traceback.print_exc()
            self._send(500, f"{type(exc).__name__}: {exc}".encode()[:2000], "text/plain")


def main():
    global BRIDGE
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/models/Xiaomi-Robotics-1-RoboCasa365")
    ap.add_argument("--port", type=int, default=10300)
    ap.add_argument("--gateway-host", default="127.0.0.1")
    ap.add_argument("--gateway-port", type=int, default=10901)
    ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--crop-ratio", type=float, default=0.95)
    ap.add_argument("--timeout", type=float, default=300.0)
    a = ap.parse_args()
    BRIDGE = Bridge(a.model, a.gateway_host, a.gateway_port, a.conns, a.crop_ratio, a.timeout)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    srv.daemon_threads = True
    print(f"vla bridge listening on 127.0.0.1:{a.port} -> gateway {a.gateway_host}:{a.gateway_port} x{a.conns}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
