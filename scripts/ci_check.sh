#!/usr/bin/env bash
set -euo pipefail

pip install -r requirements.txt
pip install pytest flake8 flake8-pyproject "black[jupyter]" nbstripout

flake8 .
black --check .
if find . -name "*.ipynb" -print -quit | grep -q .; then
  nbstripout --verify $(find . -name "*.ipynb")
else
  echo "No notebooks to check"
fi

if ls tests/*.py >/dev/null 2>&1; then
  pytest -q
else
  echo "No tests found, skipping."
fi
