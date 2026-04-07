#!/usr/bin/env bash
set -euo pipefail

cd /c/Users/arshi/OneDrive/Desktop/yes
export PATH='/c/Users/arshi/OneDrive/Desktop/yes/.tmp/validator-tests/bin-no-openenv:/usr/bin:/bin'
export MOCK_LOG_DIR='/c/Users/arshi/OneDrive/Desktop/yes/.tmp/validator-tests/logs/missing-openenv'
mkdir -p "$MOCK_LOG_DIR"

bash ./validate-submission.sh example.com .
