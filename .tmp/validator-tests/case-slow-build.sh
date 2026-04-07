#!/usr/bin/env bash
set -euo pipefail

cd /c/Users/arshi/OneDrive/Desktop/yes
export MOCK_DOCKER_INFO=ok
export MOCK_DOCKER_BUILD=ok
export MOCK_DOCKER_BUILD_SLEEP=5
export MOCK_OPENENV_VALIDATE=ok
export MOCK_CURL_MODE=200

bash ./.tmp/validator-tests/run-case.sh slow-build example.com .
