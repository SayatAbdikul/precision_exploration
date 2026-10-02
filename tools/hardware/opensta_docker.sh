#!/usr/bin/env bash
# OpenSTA shim: runs the STA engine embedded in OpenROAD (the image ships no
# standalone `sta`) inside a digest-pinned image, as the host user.
# Usage: opensta_docker.sh -version | opensta_docker.sh script.tcl
set -euo pipefail
IMAGE="openroad/orfs@sha256:6da005d1c3447799f7401215174196c75ba9ec7e28417624d0c20225f85d594c"
OPENROAD=/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
REPO="${REPO_ROOT:-/home/maveric/precision_exploration}"
PDK="${PDK_ROOT:-/home/maveric/nursultan/texer.ai/ecc/chipcompiler/thirdparty/icsprout55-pdk}"
if [[ "${1:-}" == "-version" || "${1:-}" == "--version" ]]; then
  echo "OpenSTA 3.1.0 (embedded in OpenROAD, git describe unknown) image ${IMAGE#*@}"
  exit 0
fi
# OpenROAD's link_design needs a technology; read the (read-only) LEFs first so
# that the unmodified flow script can then run as if under standalone sta.
TECH_LEF_DEFAULT="$PDK/prtech/techLEF/N551P6M.lef"
CELLS_LEF_DEFAULT="$PDK/IP/STD_cell/ics55_LLSC_H7C_V1p10C100/ics55_LLSC_H7CR/lef/ics55_LLSC_H7CR.lef"
PRE="$(mktemp)"; trap 'rm -f "$PRE"' EXIT; chmod 644 "$PRE"
cat >"$PRE" <<TCL
read_lef ${TECH_LEF:-$TECH_LEF_DEFAULT}
read_lef ${CELLS_LEF:-$CELLS_LEF_DEFAULT}
if {[info commands remove_from_collection] eq ""} {
  proc remove_from_collection {a b} {
    set r {}
    foreach x \$a { if {[lsearch -exact \$b \$x] < 0} { lappend r \$x } }
    return \$r
  }
}
source $1
TCL
envs=()
for v in TOP CLOCK_PORT LIBERTY_PATH NETLIST_PATH TARGET_PERIOD_NS REPORT_PATH \
         NETLIST LIBERTY TECH_LEF CELLS_LEF OUTPUT_DEF; do
  if [[ -n "${!v+x}" ]]; then envs+=(-e "$v=${!v}"); fi
done
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$REPO:$REPO" -v "$PDK:$PDK:ro" -v "$PRE:/tmp/sta_pre.tcl:ro" -w "$PWD" "${envs[@]}" \
  --entrypoint "$OPENROAD" "$IMAGE" -no_splash -exit /tmp/sta_pre.tcl
