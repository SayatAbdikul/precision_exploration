"""Regenerate public integer MAC requirements and oracle-driven RTL evidence.

Usage: python -m tools.hardware.integer_mac_baseline requirements|conformance
"""
from __future__ import annotations

import hashlib
import argparse
import json
import random
import subprocess
from decimal import Decimal
from pathlib import Path

from public.generic_rtl.mac.requirements import ROOT, requirements
from public.formats.oracle.manifest import manifest_sha256
from public.inference.reference.arithmetic import format_named, model_c

RTL = ROOT / "public/generic_rtl/mac/integer_mac.sv"
HARNESS = ROOT / "public/generic_rtl/mac/integer_mac_harness.sv"
OUT = ROOT / "artifacts/rtl/integer-mac-v1"
SUMMARY = ROOT / "results/summaries/integer-mac-conformance-v1.json"
REQ_OUT = ROOT / "results/summaries/hardware-requirements-v1.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_version(command: list[str]) -> str:
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return (result.stdout or result.stderr).splitlines()[0]


def word(*fields: tuple[int, int]) -> str:
    value = 0
    width = 0
    for n, bits in fields:
        value = (value << bits) | (n & ((1 << bits) - 1))
        width += bits
    return f"{value:0{(width + 3) // 4}x}"


def generate_vectors(wbits: int, abits: int, accbits: int):
    wfmt, afmt = format_named(f"int{wbits}"), format_named(f"int{abits}")
    acc = format_named(f"int{accbits}_accumulator")
    seed = 20260927 + 100 * wbits + 10 * abits + accbits
    rng = random.Random(seed)
    rows, cases = [], {"exhaustive_pairs": 0, "sampled_pairs": 0, "directed_dots": 0,
                       "valid_bubbles": 0, "resets": 0, "accumulator_overflow_dots": 0}
    mask = (1 << accbits) - 1

    def push(reset=0, valid=0, start=0, end=0, w=0, a=0, bias=0, expected_valid=0, expected=0):
        rows.append(word((reset, 1), (valid, 1), (start, 1), (end, 1), (w, wbits),
                         (a, abits), (bias, accbits), (expected_valid, 1), (expected, accbits)))

    def add_dot(pairs, bias=0):
        decoded = [(afmt.decode(a), wfmt.decode(w)) for w, a in pairs]
        expected = model_c(decoded, acc, bias=acc.decode(bias))
        for k, (w, a) in enumerate(pairs):
            push(valid=1, start=int(k == 0), end=int(k == len(pairs) - 1), w=w, a=a,
                 bias=bias if k == len(pairs) - 1 else 0,
                 expected_valid=int(k == len(pairs) - 1), expected=expected if k == len(pairs) - 1 else 0)
        cases["directed_dots"] += 1

    push(reset=1); cases["resets"] += 1
    if max(wbits, abits) <= 6:
        pairs = [(w, a) for w in range(1 << wbits) for a in range(1 << abits)]
        cases["exhaustive_pairs"] = len(pairs)
    else:
        edges_w = [0, 1, (1 << (wbits-1))-1, 1 << (wbits-1), (1 << wbits)-1]
        edges_a = [0, 1, (1 << (abits-1))-1, 1 << (abits-1), (1 << abits)-1]
        pairs = [(w, a) for w in edges_w for a in edges_a]
        pairs += [(rng.randrange(1 << wbits), rng.randrange(1 << abits)) for _ in range(4096)]
        cases["sampled_pairs"] = len(pairs)
    for idx, pair in enumerate(pairs):
        add_dot([pair])
        if idx and idx % 257 == 0:
            push(); cases["valid_bubbles"] += 1

    # Exact oracle computes both bias storage rounding and the post-reduction
    # bias add. The RTL receives the resulting stored bias code.
    bias_codes = [0, acc.encode(Decimal("0.5")), acc.encode(Decimal("1.5")),
                  acc.encode(Decimal("-1.5")), (1 << (accbits - 1)) - 1,
                  1 << (accbits - 1)]
    extreme_w = [0, 1, (1 << (wbits - 1)) - 1, 1 << (wbits - 1), (1 << wbits) - 1]
    extreme_a = [0, 1, (1 << (abits - 1)) - 1, 1 << (abits - 1), (1 << abits) - 1]
    for count in [1, 2, 3, 7, 17, 64]:
        for bc in bias_codes:
            seq = [(rng.choice(extreme_w), rng.choice(extreme_a)) for _ in range(count)]
            add_dot(seq, bc)
            push(); cases["valid_bubbles"] += 1
    add_dot([(1, 1)], bias_codes[-2])  # ACC_MAX + 1 clamps high.
    add_dot([((1 << wbits) - 1, 1)], bias_codes[-1])  # ACC_MIN - 1 clamps low.
    # Reset discards an unfinished dot, then the next start begins at zero.
    push(valid=1, start=1, w=extreme_w[-1], a=extreme_a[-1])
    push(reset=1); cases["resets"] += 1
    add_dot([(extreme_w[-1], extreme_a[-1])], bias_codes[-1])
    # INT8/INT32 can actually hit accumulator saturation in a bounded run.
    if (wbits, abits, accbits) == (8, 8, 32):
        add_dot([(128, 128)] * 131073)
        cases["accumulator_overflow_dots"] = 1
    push(); push(); cases["valid_bubbles"] += 2
    return rows, cases, seed


def testbench(wbits: int, abits: int, accbits: int, n: int) -> str:
    total = 5 + wbits + abits + 2 * accbits
    return f"""module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg reset = 1, input_valid = 0, dot_start = 0, dot_end = 0;
  reg signed [{wbits-1}:0] weight = 0;
  reg signed [{abits-1}:0] activation = 0;
  reg signed [{accbits-1}:0] bias = 0;
  wire output_valid;
  wire signed [{accbits-1}:0] result;
  reg expected_valid;
  reg [{accbits-1}:0] expected_result;
  reg [{total-1}:0] vectors [0:{n-1}];
  integer i;
  integer checks = 0;
  integer reset_checks = 0;
  integer bubble_checks = 0;
  integer result_checks = 0;
  integer_mac #(.W_BITS({wbits}), .A_BITS({abits}), .ACC_BITS({accbits})) dut (
    .clk(clk), .reset(reset), .input_valid(input_valid), .dot_start(dot_start),
    .dot_end(dot_end), .weight(weight), .activation(activation), .bias(bias),
    .output_valid(output_valid), .result(result));
  initial begin
    $readmemh("vectors.hex", vectors);
    for (i = 0; i < {n}; i = i + 1) begin
      @(negedge clk);
      {{reset, input_valid, dot_start, dot_end, weight, activation, bias,
        expected_valid, expected_result}} = vectors[i];
      @(posedge clk); #1;
      if (output_valid !== expected_valid)
        $fatal(1, "valid mismatch cycle=%0d got=%b expected=%b", i, output_valid, expected_valid);
      if (expected_valid && result !== expected_result)
        $fatal(1, "result mismatch cycle=%0d got=%h expected=%h", i, result, expected_result);
      checks = checks + 1;
      if (reset) reset_checks = reset_checks + 1;
      if (!input_valid && !reset) bubble_checks = bubble_checks + 1;
      if (expected_valid) result_checks = result_checks + 1;
    end
    $display("PASS checks=%0d results=%0d resets=%0d bubbles=%0d", checks, result_checks,
             reset_checks, bubble_checks);
    $finish;
  end
endmodule
"""


def harness_testbench(wbits: int, abits: int, accbits: int, n: int) -> str:
    total = 5 + wbits + abits + 2 * accbits
    return f"""module tb;
  reg clk = 0; always #5 clk = ~clk;
  reg reset = 1, input_valid = 0, dot_start = 0, dot_end = 0;
  reg signed [{wbits-1}:0] weight = 0;
  reg signed [{abits-1}:0] activation = 0;
  reg signed [{accbits-1}:0] bias = 0;
  wire output_valid;
  wire signed [{accbits-1}:0] result;
  reg expected_valid;
  reg [{accbits-1}:0] expected_result;
  reg [{total-1}:0] vectors [0:{n-1}];
  reg [{accbits}:0] expected [0:{n-1}];
  integer i, checks = 0, result_checks = 0;
  integer_mac_harness #(.W_BITS({wbits}), .A_BITS({abits}), .ACC_BITS({accbits})) dut (
    .clk(clk), .reset(reset), .input_valid(input_valid), .dot_start(dot_start),
    .dot_end(dot_end), .weight(weight), .activation(activation), .bias(bias),
    .output_valid(output_valid), .result(result));
  initial begin
    $readmemh("vectors.hex", vectors);
    $readmemh("harness_expected.hex", expected);
    for (i = 0; i < {n}; i = i + 1) begin
      @(negedge clk);
      {{reset, input_valid, dot_start, dot_end, weight, activation, bias,
        expected_valid, expected_result}} = vectors[i];
      @(posedge clk); #1;
      if (output_valid !== expected[i][{accbits}])
        $fatal(1, "harness valid mismatch cycle=%0d", i);
      if (expected[i][{accbits}] && result !== expected[i][{accbits-1}:0])
        $fatal(1, "harness result mismatch cycle=%0d got=%h expected=%h", i,
               result, expected[i][{accbits-1}:0]);
      checks = checks + 1;
      if (expected[i][{accbits}]) result_checks = result_checks + 1;
    end
    $display("PASS harness checks=%0d results=%0d", checks, result_checks);
    $finish;
  end
endmodule
"""


def run_harness(wbits: int, abits: int, accbits: int, rows: list[str], folder: Path):
    # Launch, DUT and capture are three sequential edge boundaries: a result
    # associated with input edge N appears at the capture after edge N+2.
    expected = []
    pending = [(0, 0), (0, 0)]
    total = 5 + wbits + abits + 2 * accbits
    for row in rows:
        value = int(row, 16)
        reset = (value >> (total - 1)) & 1
        current = ((value >> accbits) & 1, value & ((1 << accbits) - 1))
        if reset:
            observed = (0, 0)
            pending = [(0, 0), (0, 0)]
        else:
            observed = pending.pop(0)
            pending.append(current)
        expected.append(word((observed[0], 1), (observed[1], accbits)))
    (folder / "harness_expected.hex").write_text("\n".join(expected) + "\n")
    tb = folder / "harness_tb.sv"
    tb.write_text(harness_testbench(wbits, abits, accbits, len(rows)))
    compiler = subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(folder / "harness_simv"),
                               str(RTL), str(HARNESS), str(tb)], text=True, capture_output=True)
    if compiler.returncode:
        raise RuntimeError(f"harness iverilog: {compiler.stderr}")
    sim = subprocess.run(["vvp", "harness_simv"], cwd=folder, text=True, capture_output=True, timeout=180)
    if sim.returncode or not sim.stdout.startswith("PASS harness"):
        raise RuntimeError(f"harness vvp: {sim.stdout}\n{sim.stderr}")
    return sim.stdout.splitlines()[0]


def emit_variant(wbits: int, abits: int, accbits: int, folder: Path) -> Path:
    """A concrete, synthesis-ready top whose dimensions are fixed by its name."""
    top = folder / "integer_mac_variant.sv"
    top.write_text(f"""// Generated by tools.hardware.integer_mac_baseline, schema 1.0.0.
// Include integer_mac.sv in the synthesis source list.
module integer_mac_variant (
    input wire clk, reset, input_valid, dot_start, dot_end,
    input wire signed [{wbits-1}:0] weight,
    input wire signed [{abits-1}:0] activation,
    input wire signed [{accbits-1}:0] bias,
    output wire output_valid,
    output wire signed [{accbits-1}:0] result
);
    integer_mac #(.W_BITS({wbits}), .A_BITS({abits}), .ACC_BITS({accbits})) dut (
        .clk(clk), .reset(reset), .input_valid(input_valid),
        .dot_start(dot_start), .dot_end(dot_end),
        .weight(weight), .activation(activation), .bias(bias),
        .output_valid(output_valid), .result(result));
endmodule
""")
    return top


def run_one(wbits: int, abits: int, accbits: int, *, output_dir: Path = OUT):
    name = f"int{wbits}xint{abits}-acc{accbits}"
    folder = Path(output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    variant = emit_variant(wbits, abits, accbits, folder)
    variant_compile = subprocess.run(["iverilog", "-g2012", "-tnull", "-s", "integer_mac_variant",
                                      str(RTL), str(variant)], text=True, capture_output=True)
    if variant_compile.returncode:
        raise RuntimeError(f"{name}: generated variant: {variant_compile.stderr}")
    rows, cases, seed = generate_vectors(wbits, abits, accbits)
    vectors = folder / "vectors.hex"
    vectors.write_text("\n".join(rows) + "\n")
    tb = folder / "tb.sv"
    tb.write_text(testbench(wbits, abits, accbits, len(rows)))
    compiler = subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(folder / "simv"),
                               str(RTL), str(tb)], text=True, capture_output=True)
    if compiler.returncode:
        raise RuntimeError(f"{name}: iverilog: {compiler.stderr}")
    sim = subprocess.run(["vvp", "simv"], cwd=folder, text=True, capture_output=True, timeout=180)
    if sim.returncode or not sim.stdout.startswith("PASS "):
        raise RuntimeError(f"{name}: vvp: {sim.stdout}\n{sim.stderr}")
    harness_stdout = run_harness(wbits, abits, accbits, rows, folder) if (wbits, abits, accbits) == (4, 4, 32) else None
    generated_variant = (str(variant.relative_to(ROOT)) if variant.is_relative_to(ROOT)
                         else str(variant))
    return {"configuration": name, "weight_bits": wbits, "activation_bits": abits,
            "accumulator_bits": accbits, "accumulator_manifest": f"int{accbits}_accumulator",
            "manifest_sha256": {"weight": manifest_sha256(format_named(f"int{wbits}").manifest),
                                "activation": manifest_sha256(format_named(f"int{abits}").manifest),
                                "accumulator": manifest_sha256(format_named(f"int{accbits}_accumulator").manifest)},
            "scale": "1", "mac_model": "C", "reduction": "sequential",
            "cases": cases, "cycles_checked": len(rows), "result_checks": int(sim.stdout.split("results=")[1].split()[0]),
            "status": "pass", "simulation_stdout": sim.stdout.splitlines()[0], "seed": seed,
            "vectors_sha256": sha(vectors), "testbench_sha256": sha(tb),
            "generated_variant": generated_variant, "generated_variant_sha256": sha(variant),
            "generated_variant_compile": "pass",
            "harness_simulation_stdout": harness_stdout}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    req = sub.add_parser("requirements")
    req.add_argument("--output", type=Path, default=REQ_OUT)
    conf = sub.add_parser("conformance")
    conf.add_argument("--output-dir", type=Path, default=OUT)
    conf.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args(argv)
    if args.command == "requirements":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(requirements(), indent=2, sort_keys=True) + "\n")
        print(args.output)
        return
    configs = [(b, b, a) for b in (4, 5, 6, 8) for a in (32, 64)] + [(4, 8, 32), (8, 4, 32)]
    results = []
    for config in configs:
        result = run_one(*config, output_dir=args.output_dir)
        results.append(result)
        print(result["configuration"], result["simulation_stdout"], flush=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"schema_version": "integer-mac-conformance-1.0.0",
        "rtl_sha256": sha(RTL), "harness_sha256": sha(HARNESS),
        "generator_sha256": sha(Path(__file__)),
        "oracle": "public.inference.reference.arithmetic.model_c",
        "oracle_source_sha256": sha(Path(model_c.__code__.co_filename)),
        "simulator_versions": {"iverilog": tool_version(["iverilog", "-V"]),
                               "vvp": tool_version(["vvp", "-V"])},
        "scope": "raw signed integer codes at scale=1; bias supplied as stored accumulator code",
        "results": results}, indent=2, sort_keys=True) + "\n")
    print(args.summary)


if __name__ == "__main__":
    main()
