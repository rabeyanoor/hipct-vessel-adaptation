#!/bin/bash
# Uploads the project code as a new version of the Kaggle dataset rabeya100x/hip-vesseladapt-code.
# Usage: kaggle_kernels/upload_code.sh "<version message>"   (needs Kaggle credentials in the environment)
set -euo pipefail
cd "$(dirname "$0")/.."
STAGE=$(mktemp -d)
mkdir -p "$STAGE/data"
cp ./*.py "$STAGE/"
cp -r models utils losses benchmark "$STAGE/"
cp data/kidney_metadata_verified.csv "$STAGE/data/"
find "$STAGE" -name __pycache__ -type d -prune -exec rm -r {} +
echo '{"title":"hip-vesseladapt-code","id":"rabeya100x/hip-vesseladapt-code","licenses":[{"name":"CC0-1.0"}]}' > "$STAGE/dataset-metadata.json"
venv/bin/kaggle datasets version -p "$STAGE" -m "$1" --dir-mode zip
