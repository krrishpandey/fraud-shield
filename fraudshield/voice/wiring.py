"""Builds the VerifyService from config/app.yaml `verify:` and the environment (.env)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

from fraudshield.voice.bolna import BolnaClient
from fraudshield.voice.owners import OwnerDirectory
from fraudshield.voice.service import VerifyConfig, VerifyService

ROOT = Path(__file__).resolve().parents[2]


def build_verify_service(cfg: dict[str, Any], resolve: Callable[[str], tuple[dict, Any]], costs: Any,
                         runner: Any = None, env: dict[str, str] | None = None) -> VerifyService:
    env = env if env is not None else dict(os.environ)
    vc = dict(cfg.get("verify") or {})
    owners_path = vc.pop("owners_path", None)
    vc.pop("scenarios_path", None)
    conf = VerifyConfig.from_sources(vc, env)
    key = (env.get("BOLNA_API_KEY") or "").strip()
    client = BolnaClient(key) if key else None
    path = Path(owners_path) if owners_path else None
    if path is not None and not path.is_absolute():
        path = ROOT / path
    base = (env.get("VERIFY_PUBLIC_URL") or "").rstrip("/")
    hook = f"{base}/voice/webhook" if base else None
    return VerifyService(resolve, conf, OwnerDirectory.load(path, env), client=client, runner=runner, costs=costs,
                         webhook_url=hook)
