"""Oracle conformance for exact static power-of-two integer output conversion.

Usage: python -m tools.hardware.integer_output_conversion conformance [--output-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from public.formats.oracle.manifest import manifest_sha256
from public.formats.oracle.number_format import NumberFormat
from public.inference.reference.arithmetic import encode, format_named, pow2
from tools.hardware.integer_mac_baseline import ROOT, generate_vectors, sha, tool_version, word

STAGE = ROOT / "public/generic_rtl/mac/integer_output_convert.sv"
MAC = ROOT / "public/generic_rtl/mac/integer_mac.sv"
COMPOSED = ROOT / "public/generic_rtl/mac/integer_mac_scaled.sv"
HARNESS = ROOT / "public/generic_rtl/mac/integer_mac_scaled_harness.sv"
DEFAULT_OUT = ROOT / "artifacts/rtl/integer-output-conversion-v1"
DEFAULT_SUMMARY = ROOT / "results/summaries/integer-output-conversion-v1.json"


@dataclass(frozen=True)
class OutputConfig:
    accumulator_bits: int
    output_bits: int
    scale_exponent: int
    relu: bool = False

    def __post_init__(self):
        if type(self.accumulator_bits) is not int or self.accumulator_bits not in (32, 64):
            raise ValueError("accumulator width must be INT32 or INT64")
        if type(self.output_bits) is not int or self.output_bits not in (4, 5, 6, 8):
            raise ValueError("output width must be INT4, INT5, INT6 or INT8")
        if type(self.scale_exponent) is not int or not -self.accumulator_bits <= self.scale_exponent <= self.output_bits:
            raise ValueError("scale exponent outside exactly supported range")
        if type(self.relu) is not bool:
            raise ValueError("relu must be boolean")

    @classmethod
    def from_ratio(cls, accumulator_bits: int, output_bits: int, scale,
                   relu: bool = False) -> "OutputConfig":
        if isinstance(scale, bool):
            raise ValueError("scale must be a positive exact rational")
        ratio = Fraction(scale)
        if ratio <= 0:
            raise ValueError("scale must be positive")
        if ratio.numerator == 1 and ratio.denominator & (ratio.denominator - 1) == 0:
            exponent = -(ratio.denominator.bit_length() - 1)
        elif ratio.denominator == 1 and ratio.numerator & (ratio.numerator - 1) == 0:
            exponent = ratio.numerator.bit_length() - 1
        else:
            raise ValueError("arbitrary rational scale has no exact hardware mapping here")
        return cls(accumulator_bits, output_bits, exponent, relu)

    def identity(self) -> dict:
        acc = format_named(f"int{self.accumulator_bits}_accumulator")
        out = format_named(f"int{self.output_bits}")
        return {"schema_version": "integer-output-conversion-1.0.0",
                "accumulator_manifest": acc.name, "accumulator_manifest_sha256": manifest_sha256(acc.manifest),
                "output_manifest": out.name, "output_manifest_sha256": manifest_sha256(out.manifest),
                "scale_kind": "static_power_of_two", "scale_exponent": self.scale_exponent,
                "scale_ratio": str(pow2(self.scale_exponent)),
                "rounding": "rne", "overflow": "saturate", "activation": "relu" if self.relu else "identity",
                "bias_point": "already_stored_and_added_after_Model_C_reduction_before_conversion",
                "output_store": "signed_integer_code", "network_quality_attached": False}


def oracle_code(acc_code: int, config: OutputConfig) -> int:
    acc = format_named(f"int{config.accumulator_bits}_accumulator")
    out = format_named(f"int{config.output_bits}")
    value = acc.decode(acc_code) * pow2(config.scale_exponent)
    if config.relu and value < 0:
        value = 0
    return encode(out, value)


def stage_vectors(config: OutputConfig):
    width = config.accumulator_bits
    mask = (1 << width) - 1
    out_max = (1 << (config.output_bits - 1)) - 1
    values = [0, 1, 2, 3, 4, 5, 6, 7, 8, out_max - 1, out_max, out_max + 1,
              2 * out_max + 1, (1 << (width - 1)) - 1, -(1 << (width - 1))]
    values += [-v for v in values if v]
    rng = random.Random(20260927 + width * 100 + config.output_bits * 10 + config.scale_exponent + int(config.relu))
    values += [rng.randrange(-(1 << (width - 1)), 1 << (width - 1)) for _ in range(256)]
    rows = []
    rows.append(word((1, 1), (0, 1), (0, width), (0, 1), (0, config.output_bits)))
    expected_results = 0
    for index, signed in enumerate(values):
        code = signed & mask
        expected = oracle_code(code, config)
        rows.append(word((0, 1), (1, 1), (code, width), (1, 1), (expected, config.output_bits)))
        expected_results += 1
        if index % 19 == 0:
            rows.append(word((0, 1), (0, 1), (0, width), (0, 1), (0, config.output_bits)))
    rows.append(word((1, 1), (0, 1), (0, width), (0, 1), (0, config.output_bits)))
    rows.append(word((0, 1), (0, 1), (0, width), (0, 1), (0, config.output_bits)))
    return rows, expected_results


def stage_tb(config: OutputConfig, count: int) -> str:
    a, o = config.accumulator_bits, config.output_bits
    return f"""module tb;
  reg clk=0; always #5 clk=~clk;
  reg reset=0, input_valid=0;
  reg signed [{a-1}:0] accumulator_code=0;
  wire output_valid;
  wire signed [{o-1}:0] output_code;
  reg expected_valid;
  reg [{o-1}:0] expected_code;
  reg [{a+o+2}:0] vectors [0:{count-1}];
  integer i, checks=0;
  integer_output_convert #(.ACC_BITS({a}), .OUT_BITS({o}),
    .SCALE_EXP({config.scale_exponent}), .RELU({int(config.relu)})) dut (
    .clk(clk), .reset(reset), .input_valid(input_valid),
    .accumulator_code(accumulator_code), .output_valid(output_valid), .output_code(output_code));
  initial begin
    $readmemh("vectors.hex", vectors);
    for (i=0; i<{count}; i=i+1) begin
      @(negedge clk);
      {{reset, input_valid, accumulator_code, expected_valid, expected_code}} = vectors[i];
      @(posedge clk); #1;
      if (output_valid !== expected_valid) $fatal(1, "valid mismatch %0d", i);
      if (expected_valid && output_code !== expected_code)
        $fatal(1, "code mismatch %0d got=%h expected=%h", i, output_code, expected_code);
      checks=checks+1;
    end
    $display("PASS stage checks=%0d", checks); $finish;
  end
endmodule
"""


def _simulate(folder: Path, tb_name: str, sources: list[Path], expected_prefix: str):
    compilation = subprocess.run(["iverilog", "-g2012", "-s", "tb", "-o", str(folder / "simv"),
                                  *(str(source) for source in sources), str(folder / tb_name)],
                                 capture_output=True, text=True)
    if compilation.returncode:
        raise RuntimeError(compilation.stderr)
    run = subprocess.run(["vvp", "simv"], cwd=folder, capture_output=True, text=True, timeout=180)
    if run.returncode or not run.stdout.startswith(expected_prefix):
        raise RuntimeError(run.stdout + "\n" + run.stderr)
    return run.stdout.splitlines()[0]


def run_stage(config: OutputConfig, output_dir: Path):
    name = f"acc{config.accumulator_bits}-out{config.output_bits}-e{config.scale_exponent}-relu{int(config.relu)}"
    folder = Path(output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    rows, expected_results = stage_vectors(config)
    vectors = folder / "vectors.hex"
    vectors.write_text("\n".join(rows) + "\n")
    tb = folder / "tb.sv"
    tb.write_text(stage_tb(config, len(rows)))
    identity = config.identity()
    (folder / "config.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    stdout = _simulate(folder, "tb.sv", [STAGE], "PASS stage")
    return {"configuration": identity, "status": "pass", "cycles_checked": len(rows),
            "result_checks": expected_results, "simulation_stdout": stdout,
            "vectors_sha256": sha(vectors), "testbench_sha256": sha(tb),
            "config_sha256": sha(folder / "config.json")}


def integrated_expected(raw_rows: list[str], wbits: int, abits: int, config: OutputConfig):
    acc = config.accumulator_bits
    total = 5 + wbits + abits + 2 * acc
    pending = (0, 0)
    rows = []
    for raw in raw_rows:
        value = int(raw, 16)
        reset = (value >> (total - 1)) & 1
        current = ((value >> acc) & 1, value & ((1 << acc) - 1))
        if reset:
            observed = (0, 0)
            pending = (0, 0)
        else:
            observed = pending
            pending = current
        result = oracle_code(observed[1], config) if observed[0] else 0
        rows.append(word((observed[0], 1), (result, config.output_bits)))
    return rows


def integrated_tb(wbits: int, abits: int, config: OutputConfig, count: int):
    acc, out = config.accumulator_bits, config.output_bits
    total = 5 + wbits + abits + 2 * acc
    return f"""module tb;
  reg clk=0; always #5 clk=~clk;
  reg reset=0, input_valid=0, dot_start=0, dot_end=0;
  reg signed [{wbits-1}:0] weight=0;
  reg signed [{abits-1}:0] activation=0;
  reg signed [{acc-1}:0] bias=0;
  reg ignored_valid;
  reg [{acc-1}:0] ignored_accumulator;
  wire output_valid;
  wire signed [{out-1}:0] result;
  reg [{total-1}:0] input_vectors [0:{count-1}];
  reg [{out}:0] expected [0:{count-1}];
  integer i, result_checks=0;
  integer_mac_scaled #(.W_BITS({wbits}), .A_BITS({abits}), .ACC_BITS({acc}),
    .OUT_BITS({out}), .SCALE_EXP({config.scale_exponent}), .RELU({int(config.relu)})) dut (
    .clk(clk), .reset(reset), .input_valid(input_valid), .dot_start(dot_start),
    .dot_end(dot_end), .weight(weight), .activation(activation), .bias(bias),
    .output_valid(output_valid), .result(result));
  initial begin
    $readmemh("input.hex", input_vectors);
    $readmemh("expected.hex", expected);
    for (i=0; i<{count}; i=i+1) begin
      @(negedge clk);
      {{reset,input_valid,dot_start,dot_end,weight,activation,bias,
        ignored_valid,ignored_accumulator}} = input_vectors[i];
      @(posedge clk); #1;
      if (output_valid !== expected[i][{out}]) $fatal(1, "integrated valid mismatch %0d", i);
      if (expected[i][{out}] && result !== expected[i][{out-1}:0])
        $fatal(1, "integrated result mismatch %0d got=%h expected=%h", i, result, expected[i][{out-1}:0]);
      if (expected[i][{out}]) result_checks=result_checks+1;
    end
    $display("PASS integrated checks=%0d results=%0d", {count}, result_checks); $finish;
  end
endmodule
"""


def run_integrated(wbits: int, abits: int, config: OutputConfig, output_dir: Path):
    name = f"mac{wbits}x{abits}-acc{config.accumulator_bits}-out{config.output_bits}-e{config.scale_exponent}-relu{int(config.relu)}"
    folder = Path(output_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    raw, cases, seed = generate_vectors(wbits, abits, config.accumulator_bits)
    expected = integrated_expected(raw, wbits, abits, config)
    (folder / "input.hex").write_text("\n".join(raw) + "\n")
    (folder / "expected.hex").write_text("\n".join(expected) + "\n")
    tb = folder / "tb.sv"
    tb.write_text(integrated_tb(wbits, abits, config, len(raw)))
    identity = {"schema_version": "integer-mac-scaled-1.0.0", "weight_manifest": f"int{wbits}",
                "weight_manifest_sha256": manifest_sha256(format_named(f"int{wbits}").manifest),
                "activation_manifest": f"int{abits}",
                "activation_manifest_sha256": manifest_sha256(format_named(f"int{abits}").manifest),
                "mac_model": "C", "reduction": "sequential",
                "conversion": config.identity(), "network_quality_attached": False}
    (folder / "config.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    stdout = _simulate(folder, "tb.sv", [MAC, STAGE, COMPOSED], "PASS integrated")
    return {"configuration": identity, "status": "pass", "cases": cases, "seed": seed,
            "cycles_checked": len(raw), "result_checks": int(stdout.split("results=")[1]),
            "simulation_stdout": stdout, "input_sha256": sha(folder / "input.hex"),
            "expected_sha256": sha(folder / "expected.hex"), "testbench_sha256": sha(tb),
            "config_sha256": sha(folder / "config.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("conformance",))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args(argv)
    rows = []
    for acc in (32, 64):
        for out in (4, 5, 6, 8):
            for exponent in (-acc, -1, 0, 1, out):
                for relu in (False, True):
                    rows.append(run_stage(OutputConfig(acc, out, exponent, relu), args.output_dir))
    integrated = [run_integrated(4, 4, OutputConfig(32, 4, -1), args.output_dir),
                  run_integrated(6, 6, OutputConfig(32, 6, 2, True), args.output_dir),
                  run_integrated(8, 8, OutputConfig(64, 8, -2), args.output_dir)]
    summary = {"schema_version": "integer-output-conversion-conformance-1.0.0",
               "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (STAGE, MAC, COMPOSED, HARNESS)},
               "oracle_source": "public.formats.oracle.number_format.NumberFormat.encode and public.inference.reference.arithmetic.model_c",
               "oracle_source_hashes": {
                   "number_format": sha(Path(NumberFormat.__init__.__code__.co_filename)),
                   "reference_arithmetic": sha(ROOT / "public/inference/reference/arithmetic.py")},
               "simulator_versions": {"iverilog": tool_version(["iverilog", "-V"]),
                                      "vvp": tool_version(["vvp", "-V"])},
               "generator_sha256": sha(Path(__file__)),
               "stage_results": rows, "integrated_results": integrated,
               "support_scope": "static exact power-of-two scale only; no arbitrary rational scale, no network quality"}
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"PASS {len(rows)} stage configs, {len(integrated)} integrated configs -> {args.summary}")


if __name__ == "__main__":
    main()
