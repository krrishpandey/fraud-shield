"""python -m fraudshield.audit verify --path audit.jsonl"""
import argparse
import json
import sys

from fraudshield.audit.log import verify_file


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fraudshield.audit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--path", required=True)
    args = ap.parse_args(argv)
    res = verify_file(args.path)
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
