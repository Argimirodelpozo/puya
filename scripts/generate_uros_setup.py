#!/usr/bin/env python3
"""Pre-compile the uros-relay library to embedded AWST.

The source of truth is `src/_uros_lib/` -- the generic `setup` helper (`setup.py`) and the
`infra` contract whose methods the splitter splices into the user's `main` (`infra.py`). This
compiles them to AWST and embeds them in the puya package at
`src/puya/ir/uros/lib.awst.json`, so the compiler can merge + emit `setup` and
graft the infra without re-parsing source. Mirrors `generate_puya_lib.py`; run via
`poe gen_uros_setup` (part of `poe gen`).
"""
import subprocess
import sys

from scripts.script_utils import VCS_ROOT

LIB_NAME = "_uros_lib"


def main() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "puyapy", "--output-awst-json",
            "--no-output-teal", "--no-output-source-map", "--no-output-arc56",
            f"src/{LIB_NAME}",
        ],
        check=True,
        cwd=VCS_ROOT,
    )
    awst_path = VCS_ROOT / "module.awst.json"
    lib_path = VCS_ROOT / "src" / LIB_NAME
    out_path = VCS_ROOT / "src" / "puya" / "ir" / "uros" / "lib.awst.json"
    awst = awst_path.read_text(encoding="utf8")
    # normalise absolute source paths to null so the embedded artifact is machine-independent
    for src in lib_path.glob("*.py"):
        path_as_str = str(src).replace("\\", "\\\\")
        awst = awst.replace(f'"file": "{path_as_str}",', '"file": null,')
    out_path.write_text(awst, encoding="utf8")
    awst_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
