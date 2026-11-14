"""`python -m co_extract FILE` -> JSON on stdout."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .pipeline import extract_file


def _load_dotenv() -> None:
    """Load ./.env; variables already set in the environment take precedence."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from pathlib import Path

    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        load_dotenv(env_file)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    ap = argparse.ArgumentParser(prog="co_extract", description="Extract a change order to validated JSON.")
    ap.add_argument("file", help="PDF or text file")
    ap.add_argument("--offline", action="store_true", help="pattern extraction only, no LLM call")
    ap.add_argument("--samples", type=int, default=int(os.environ.get("LLM_SAMPLES", "3")),
                    help="LLM samples for self-consistency (default 3)")
    ap.add_argument("--provider", choices=["openai", "anthropic"], default=None,
                    help="LLM provider (default: LLM_PROVIDER, else whichever key is set)")
    ap.add_argument("--model", default=None, help="override LLM_MODEL / ANTHROPIC_MODEL")
    ap.add_argument("--no-cache", action="store_true", help="always call the LLM")
    args = ap.parse_args(argv)

    client = None
    if not args.offline:
        from .providers import default_provider, make_client

        provider = args.provider or default_provider()
        if provider is None:
            print("error: no LLM key set (LLM_API_KEY or ANTHROPIC_API_KEY in .env), or pass --offline",
                  file=sys.stderr)
            return 2
        try:
            client = make_client(provider, model=args.model, cache=not args.no_cache)
        except RuntimeError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2

    result = extract_file(args.file, client=client, n_samples=args.samples)
    json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
