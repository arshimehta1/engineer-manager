#!/usr/bin/env bash
#
# validate-submission.sh - Enhanced OpenEnv Submission Validator
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
REQUIRED_CMDS=("curl" "docker" "openenv")

if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BLUE='' BOLD='' NC=''
fi

log()  { printf "[%s] %b\n" "$(date +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}${BOLD}PASS${NC} -- $1"; }
fail() { log "${RED}${BOLD}FAIL${NC} -- $1"; }
warn() { log "${YELLOW}${BOLD}WARN${NC} -- $1"; }
hint() { printf "  ${YELLOW}Hint:${NC} %b\n" "$1"; }

show_spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='|/-\'
  while kill -0 "$pid" 2>/dev/null; do
    local temp=${spinstr#?}
    printf " [%c]  " "$spinstr"
    spinstr=$temp${spinstr%"$temp"}
    sleep "$delay"
    printf "\b\b\b\b\b\b"
  done
  printf "    \b\b\b\b"
}

stop_at() {
  printf "\n${RED}${BOLD}Validation stopped at %s.${NC} Please address the error above.\n" "$1"
  exit 1
}

PING_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$PING_URL" ]; then
  printf "${BOLD}Usage:${NC} %s <ping_url> [repo_dir]\n" "$0"
  exit 1
fi

[[ ! $PING_URL =~ ^http ]] && PING_URL="https://$PING_URL"
PING_URL="${PING_URL%/}"

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  fail "Directory '$REPO_DIR' not found."
  exit 1
fi

printf "\n${BOLD}========================================${NC}\n"
printf "${BOLD}   OpenEnv Submission Validator v2.0    ${NC}\n"
printf "${BOLD}========================================${NC}\n"
log "Target Repo: $REPO_DIR"
log "Remote URL:  $PING_URL"
printf "\n"

log "${BOLD}Step 0/3: Checking Environment...${NC}"
for cmd in "${REQUIRED_CMDS[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    fail "Dependency missing: $cmd"
    [[ "$cmd" == "openenv" ]] && hint "Run: pip install openenv-core"
    stop_at "Pre-flight"
  fi
done

if ! docker info >/dev/null 2>&1; then
  fail "Docker daemon is not running."
  hint "Ensure Docker Desktop or the Docker Engine is active."
  stop_at "Pre-flight"
fi
pass "Environment is ready."

log "${BOLD}Step 1/3: Pinging HF Space${NC} ($PING_URL/reset) ..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
  -H "Content-Type: application/json" -d '{}' \
  "$PING_URL/reset" --max-time 20 || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
  pass "HF Space is live and responding."
else
  fail "HF Space /reset returned $HTTP_CODE"
  hint "Check if your Space is sleepy or still building."
  stop_at "Step 1"
fi

log "${BOLD}Step 2/3: Running Docker Build${NC} (This may take a while)..."
if [ -f "$REPO_DIR/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR"
elif [ -f "$REPO_DIR/server/Dockerfile" ]; then
  DOCKER_CONTEXT="$REPO_DIR/server"
else
  fail "No Dockerfile found."
  stop_at "Step 2"
fi

BUILD_LOG=$(mktemp)
(docker build -t openenv-test "$DOCKER_CONTEXT" > "$BUILD_LOG" 2>&1) &
BUILD_PID=$!
show_spinner "$BUILD_PID"
wait "$BUILD_PID"

if [ $? -eq 0 ]; then
  pass "Docker build successful."
else
  fail "Docker build failed."
  printf "${YELLOW}--- Build Error Snippet ---${NC}\n"
  tail -n 15 "$BUILD_LOG"
  hint "Full logs can be found at $BUILD_LOG"
  stop_at "Step 2"
fi

log "${BOLD}Step 3/3: Running OpenEnv Validate${NC} ..."
if (cd "$REPO_DIR" && openenv validate); then
  pass "Schema validation passed."
else
  fail "openenv validate discovered structural errors."
  stop_at "Step 3"
fi

printf "\n"
printf "${GREEN}${BOLD}========================================${NC}\n"
printf "${GREEN}${BOLD}   ALL CHECKS PASSED SUCCESSFULLY!      ${NC}\n"
printf "${GREEN}${BOLD}   Your submission is ready for HF.     ${NC}\n"
printf "${GREEN}${BOLD}========================================${NC}\n\n"

exit 0
