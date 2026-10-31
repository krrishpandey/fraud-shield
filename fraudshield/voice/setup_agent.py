"""Create or update the Tracd Verify Bolna agent from fraudshield/voice/agent_config.json, and place a test call.

    python -m fraudshield.voice.setup_agent                  create the agent (or update it if BOLNA_AGENT_ID is set)
    python -m fraudshield.voice.setup_agent --test-call +91XXXXXXXXXX   one real call with a sample booking

Reads BOLNA_API_KEY (and BOLNA_AGENT_ID, VERIFY_PUBLIC_URL, BOLNA_FROM_NUMBER) from .env. Trial accounts can only call
numbers verified in the Bolna dashboard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from fraudshield.voice.bolna import BolnaClient, BolnaError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("agent_config.json")

SAMPLE = {"company_name": "Sri Murugan Textiles", "weight_kg": 18.0, "service": "express", "origin": "Chennai",
          "destination": "Delhi", "booking_time": "2:14 am", "device_is_new": "yes", "device_note": ", from a new device",
          "decision_id": "test-call", "booking_id": "test-call", "language": "en"}


def _load_env() -> None:
    from fraudshield.explain.groq_client import load_dotenv  # noqa: PLC0415
    load_dotenv(ROOT / ".env")


def agent_body() -> tuple[dict, list[dict]]:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    body = {"agent_config": raw["agent_config"], "agent_prompts": raw["agent_prompts"]}
    public = (os.environ.get("VERIFY_PUBLIC_URL") or "").rstrip("/")
    if public:
        secret = os.environ.get("VERIFY_WEBHOOK_SECRET") or ""
        body["agent_config"]["webhook_url"] = f"{public}/voice/webhook" + (f"?token={secret}" if secret else "")
    return body, raw.get("dispositions") or []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fraudshield.voice.setup_agent")
    ap.add_argument("--test-call", metavar="PHONE", help="place one real call to this verified number (E.164)")
    args = ap.parse_args(argv)
    _load_env()
    key = (os.environ.get("BOLNA_API_KEY") or "").strip()
    if not key:
        print("BOLNA_API_KEY is missing. Put it in .env (see .env.example) and run again.")
        return 2
    client = BolnaClient(key)
    agent_id = (os.environ.get("BOLNA_AGENT_ID") or "").strip()
    try:
        if args.test_call:
            if not agent_id:
                print("BOLNA_AGENT_ID is missing: run this command without --test-call first.")
                return 2
            # English-only agent: no agent_data.language (Bolna rejects it unless the agent is multilingual)
            out = client.place_call(agent_id, args.test_call, SAMPLE, language=None,
                                    from_phone=os.environ.get("BOLNA_FROM_NUMBER") or None, bypass_guardrails=True)
            ex = out["execution_id"]
            print(f"Call queued: execution {ex}. Your phone should ring in a few seconds.")
            for _ in range(60):  # follow it for up to 3 minutes
                time.sleep(3)
                e = client.get_execution(ex)
                print(f"  status: {e.get('status')}")
                if e.get("status") in ("completed", "call-disconnected", "no-answer", "busy", "failed", "error",
                                       "canceled", "balance-low"):
                    print("Transcript:\n" + (e.get("transcript") or "(none)"))
                    print("Extracted:", json.dumps(e.get("extracted_data"), indent=1))
                    print("Cost (total_cost):", e.get("total_cost"))
                    break
            return 0
        body, dispositions = agent_body()
        if agent_id:
            client.update_agent(agent_id, body)
            print(f"Updated agent {agent_id}.")
        else:
            out = client.create_agent(body)
            agent_id = out["agent_id"]
            print(f"Created agent {agent_id}.")
            if dispositions:
                client.create_dispositions(agent_id, dispositions)
                print(f"Added {len(dispositions)} extraction questions (owner_confirmed, detail_answer, change_reason).")
            print(f"\nNow add this line to .env and restart the app:\nBOLNA_AGENT_ID={agent_id}")
        return 0
    except BolnaError as e:
        print(f"Bolna said no: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
