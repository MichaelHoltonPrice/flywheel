#!/usr/bin/env bash
set -euo pipefail

echo "=== ruff check ==="
ruff check .

echo "=== pytest ==="
python -m pytest
