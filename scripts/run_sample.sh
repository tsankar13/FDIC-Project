#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for local smoke/demo runs.
# Canonical interface remains: `fdic-ml ...` (see README).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

pip install -e . -q

fdic-ml --config backend/configs/sample.json --mode both
fdic-ml --generate-synopsis-report --output-dir backend/artifacts --reports-dir backend/artifacts/reports/primary

echo "Sample run complete. Check backend/artifacts/ and backend/artifacts/reports/primary/"
