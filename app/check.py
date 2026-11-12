"""
Print what's connected and what's still missing.

  python -m app.check          # reads the environment (and .env if python-dotenv is installed)

Exit code 0 when nothing is half-configured or still to do, 1 otherwise.
"""
import sys

MARK = {"on": "ok  ", "off": "-   ", "todo": "TODO", "problem": "FIX "}


def main():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    from . import connections
    items = connections.check()
    for i in items:
        print(f"[{MARK[i['state']]}] {i['name']}: {i['detail']}")
        if i["state"] in ("todo", "problem") and i["fix"]:
            print(f"         -> {i['fix']}")
    ok = connections.ready(items)
    print("\nReady for beta testers." if ok else "\nNot ready yet: see TODO / FIX above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
