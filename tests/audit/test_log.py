import json
import subprocess
import sys
import threading

import pytest

from fraudshield.audit.log import AuditLog, canonical_hash


def test_append_links_records_by_hash(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    seq0, h0 = log.append("decision", {"decision_id": "d1"})
    seq1, h1 = log.append("explanation", {"decision_id": "d1", "text": "x"})
    lines = [json.loads(x) for x in (tmp_path / "a.jsonl").read_text().splitlines()]
    assert (seq0, seq1) == (0, 1)
    assert lines[0]["prev_hash"] == "0" * 64
    assert lines[1]["prev_hash"] == h0
    assert lines[1]["record_hash"] == h1
    body = {k: v for k, v in lines[1].items() if k != "record_hash"}
    assert canonical_hash(body) == h1


def test_unknown_event_type_rejected(tmp_path):
    with pytest.raises(ValueError):
        AuditLog(tmp_path / "a.jsonl").append("hack", {})


def test_verify_ok_and_head(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")
    for i in range(20):
        _, h = log.append("decision", {"i": i})
    res = log.verify()
    assert res == {"ok": True, "records": 20, "head_hash": h, "first_bad_index": None}


def test_verify_empty_log(tmp_path):
    res = AuditLog(tmp_path / "a.jsonl").verify()
    assert res["ok"] is True and res["records"] == 0


def test_flip_one_byte_fails_at_that_record(tmp_path):
    p = tmp_path / "a.jsonl"
    log = AuditLog(p)
    for i in range(10):
        log.append("decision", {"i": i, "action": "allow"})
    raw = bytearray(p.read_bytes())
    lines = raw.split(b"\n")
    target = lines[6]
    idx = target.index(b"allow")
    target[idx] = ord("b")  # 'allow' -> 'bllow'
    lines[6] = target
    p.write_bytes(b"\n".join(lines))
    res = AuditLog(p).verify()
    assert res["ok"] is False
    assert res["first_bad_index"] == 6


def test_reopen_continues_chain(tmp_path):
    p = tmp_path / "a.jsonl"
    AuditLog(p).append("decision", {"i": 0})
    log2 = AuditLog(p)
    seq, _ = log2.append("outcome", {"i": 1})
    assert seq == 1
    assert log2.verify()["ok"]


def test_concurrent_appends_keep_chain_valid(tmp_path):
    log = AuditLog(tmp_path / "a.jsonl")

    def work(k):
        for i in range(25):
            log.append("analyst_feedback", {"k": k, "i": i})

    ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    res = log.verify()
    assert res["ok"] and res["records"] == 200


def test_cli_verify(tmp_path):
    p = tmp_path / "a.jsonl"
    AuditLog(p).append("decision", {"i": 0})
    out = subprocess.run([sys.executable, "-m", "fraudshield.audit", "verify", "--path", str(p)],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert json.loads(out.stdout)["ok"] is True
    p.write_text(p.read_text().replace('"i":0', '"i":1'))
    out = subprocess.run([sys.executable, "-m", "fraudshield.audit", "verify", "--path", str(p)],
                         capture_output=True, text=True)
    assert out.returncode == 1
