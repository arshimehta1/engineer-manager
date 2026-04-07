#!/usr/bin/env bash
set -euo pipefail

case_name="${1:?case name required}"
ping_url="${2:?ping url required}"
repo_dir="${3:-.}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
mock_bin="$script_dir/bin"
log_dir="$script_dir/logs/$case_name"

rm -rf "$log_dir"
mkdir -p "$log_dir"

export MOCK_LOG_DIR="$log_dir"
export PATH="$mock_bin:$PATH"

bash "$repo_root/validate-submission.sh" "$ping_url" "$repo_dir"
