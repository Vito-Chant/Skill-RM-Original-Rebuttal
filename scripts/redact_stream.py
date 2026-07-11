#!/usr/bin/env python3
from __future__ import annotations

import sys

from rebuttal_common import sanitize_text


def main() -> None:
    for line in sys.stdin:
        sys.stdout.write(sanitize_text(line))
        sys.stdout.flush()


if __name__ == "__main__":
    main()
