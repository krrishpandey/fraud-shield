"""FraudShield desktop app: runs the API in-process and shows the console in a native window.

Usage: python -m fraudshield.desktop [--laya-mode auto|local|cpu|cached] [--port N]
"""
from __future__ import annotations

import argparse
import os
import socket
import threading
import time
import urllib.request
from pathlib import Path


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def choose_laya_mode(requested: str, *, has_cuda: bool, has_laya: bool) -> str:
    """Pick how Laya runs. PCs without a GPU or without the ml extra get cached answers."""
    if not has_laya:
        return "cached"
    if requested == "auto":
        # The app decides: fine-tuned checkpoint present -> Laya decides; otherwise cached answers.
        return "auto" if has_cuda else "cached"
    return requested


def wait_for_server(url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if r.status == 200:
                    return True
        except OSError:
            time.sleep(0.2)
    return False


def load_config(mode: str) -> dict:
    """App config (FS_CONFIG or config/app.yaml) with the chosen Laya mode applied."""
    from fraudshield.api.app import load_config as load_app_config

    cfg = load_app_config(os.environ.get("FS_CONFIG"))
    cfg.setdefault("laya", {})["mode"] = mode
    return cfg


def _detect_ml() -> tuple[bool, bool]:
    try:
        import laya  # noqa: F401
        import torch
    except ImportError:
        return False, False
    return bool(torch.cuda.is_available()), True


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="fraudshield")
    ap.add_argument("--laya-mode", default=os.environ.get("FS_LAYA_MODE", "auto"),
                    choices=["auto", "local", "cpu", "cached"])
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args(argv)

    from fraudshield.explain.groq_client import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")

    has_cuda, has_laya = _detect_ml()
    mode = choose_laya_mode(args.laya_mode, has_cuda=has_cuda, has_laya=has_laya)
    os.environ["FS_LAYA_MODE"] = mode
    port = args.port or find_free_port()
    base = f"http://127.0.0.1:{port}"

    import uvicorn
    import webview

    from fraudshield.api import real_components
    from fraudshield.api.app import build_app

    real_components.warm()

    server = uvicorn.Server(uvicorn.Config(build_app(load_config(mode)), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    # Loading Laya on the GPU takes up to ~2 minutes on first run (model download), so wait generously.
    if not wait_for_server(f"{base}/health", timeout=300):
        raise SystemExit("FraudShield backend did not start; see the console output above.")

    webview.create_window("FraudShield", base, width=1360, height=860, min_size=(1024, 700))
    webview.start()
    server.should_exit = True


if __name__ == "__main__":
    main()
