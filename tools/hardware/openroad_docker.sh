#!/usr/bin/env bash
# Runs OpenROAD (and through it OpenSTA) from a digest-pinned image as the
# host user. The repository (read-write) and the ICsprout55 PDK (read-only)
# are mounted at their host absolute paths, so scripts, netlists and Liberty
# files use identical paths inside and outside the container.
#
# Usage:
#   openroad_docker.sh --version            tool versions queried from the tool
#   openroad_docker.sh [openroad args...]   e.g. openroad_docker.sh /abs/script.tcl
# `-no_splash -exit` is prepended. Every argument is forwarded verbatim (spaces
# are safe) and openroad's exit status is returned. Nothing is written outside
# the repository (--version uses a temporary file in the working directory).
# Environment: variables named in OPENROAD_DOCKER_ENV (space separated) are
# forwarded, plus TOP CLOCK_PORT LIBERTY_PATH NETLIST_PATH TARGET_PERIOD_NS
# REPORT_PATH NETLIST LIBERTY TECH_LEF CELLS_LEF OUTPUT_DEF.
# Override mount roots with REPO_ROOT and PDK_ROOT. The working directory must
# be inside REPO_ROOT. The container has no network.
set -euo pipefail
IMAGE="openroad/orfs@sha256:6da005d1c3447799f7401215174196c75ba9ec7e28417624d0c20225f85d594c"
OPENROAD=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
REPO="${REPO_ROOT:-/home/maveric/precision_exploration}"
PDK="${PDK_ROOT:-/home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk}"

[[ -d "$REPO" ]] || { echo "openroad_docker: REPO_ROOT not a directory: $REPO" >&2; exit 2; }
[[ -d "$PDK" ]] || { echo "openroad_docker: PDK_ROOT not a directory: $PDK" >&2; exit 2; }
case "$PWD/" in
  "$REPO"/*) ;;
  *) echo "openroad_docker: working directory $PWD is outside $REPO" >&2; exit 2 ;;
esac

mounts=(-v "$REPO:$REPO" -v "$PDK:$PDK:ro")
envs=()
names=(TOP CLOCK_PORT LIBERTY_PATH NETLIST_PATH TARGET_PERIOD_NS REPORT_PATH
       NETLIST LIBERTY TECH_LEF CELLS_LEF OUTPUT_DEF)
# shellcheck disable=SC2206
names+=(${OPENROAD_DOCKER_ENV:-})
for v in "${names[@]}"; do
  if [[ -n "${!v+x}" ]]; then envs+=(-e "$v=${!v}"); fi
done

if [[ "${1:-}" == "--version" || "${1:-}" == "-version" ]]; then
  VT="$(mktemp "$PWD/.openroad_version.XXXXXX")"; trap 'rm -f "$VT"' EXIT; chmod 644 "$VT"
  cat >"$VT" <<'TCL'
puts "OpenSTA [sta::version]"
puts "OpenROAD [ord::openroad_version]"
TCL
  echo "image ${IMAGE}"
  exec_args=(-v "$VT:/tmp/version.tcl:ro")
  docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp "${exec_args[@]}" \
    --entrypoint "$OPENROAD" "$IMAGE" -no_splash -exit /tmp/version.tcl
  exit $?
fi

docker run --rm --network none -u "$(id -u):$(id -g)" -e HOME=/tmp \
  "${mounts[@]}" -w "$PWD" ${envs[@]+"${envs[@]}"} \
  --entrypoint "$OPENROAD" "$IMAGE" -no_splash -exit "$@"
