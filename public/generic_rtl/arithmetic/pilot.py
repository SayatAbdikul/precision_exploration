"""Generate manifest-exact primitive ROM baselines and registered pilot wrappers.

These exhaustive lookup circuits validate semantics and flow plumbing. They
are deliberately not a claim about the best arithmetic architecture.
"""
from __future__ import annotations

from public.inference.reference.arithmetic import format_named
from public.formats.oracle.manifest import manifest_sha256


def rtl(format_name, operation, internal_stages=0):
    if operation not in {"add", "mul"} or type(internal_stages) is not int or not 0 <= internal_stages <= 3:
        raise ValueError("pilot requires ADD/MUL and 0..3 internal stages")
    fmt = format_named(format_name)
    if fmt.bits > 8:
        raise ValueError("exhaustive primitive pilot requires a <=8-bit format")
    bits, codes = fmt.bits, 1 << fmt.bits
    latency = internal_stages + 2
    rows = [f"// Manifest {fmt.name}: {manifest_sha256(fmt.manifest)}",
            "// Primitive rounded result, scale=1; not a Model C MAC.",
            f"module precision_primitive(input [{bits-1}:0] a, b, output reg [{bits-1}:0] result);",
            "always @* begin", "case ({a,b})"]
    for a in range(codes):
        for b in range(codes):
            value = getattr(fmt, operation)(a, b)
            rows.append(f"{2*bits}'d{a*codes+b}: result = {bits}'d{value};")
    rows.extend([f"default: result = {bits}'d0;", "endcase", "end", "endmodule",
        f"module precision_pilot(input clk, reset, input_valid, input [{bits-1}:0] a, b,",
        f"output output_valid, output [{bits-1}:0] result);",
        f"reg [{bits-1}:0] a_q, b_q; reg valid_q; wire [{bits-1}:0] primitive_result;",
        "precision_primitive dut(a_q, b_q, primitive_result);",
        f"reg [{bits-1}:0] data_pipe [0:{internal_stages}];",
        f"reg [{internal_stages}:0] valid_pipe; integer i;",
        f"assign result = data_pipe[{internal_stages}]; assign output_valid = valid_pipe[{internal_stages}];",
        "always @(posedge clk) begin",
        "if (reset) begin a_q <= 0; b_q <= 0; valid_q <= 0; valid_pipe <= 0;",
        f"for (i=0; i<={internal_stages}; i=i+1) data_pipe[i] <= 0;",
        "end else begin",
        "a_q <= a; b_q <= b; valid_q <= input_valid;",
        "data_pipe[0] <= primitive_result; valid_pipe[0] <= valid_q;",
        f"for (i=1; i<={internal_stages}; i=i+1) begin data_pipe[i] <= data_pipe[i-1]; valid_pipe[i] <= valid_pipe[i-1]; end",
        "end end endmodule"])
    return "\n".join(rows) + "\n", latency


def testbench(format_name, operation, latency):
    fmt = format_named(format_name)
    codes, bits = 1 << fmt.bits, fmt.bits
    rows = ["module tb; reg clk=0; always #5 clk=~clk; reg reset=1, input_valid=0;",
            f"reg [{bits-1}:0] a=0,b=0; wire [{bits-1}:0] result; wire output_valid;",
            "precision_pilot dut(clk,reset,input_valid,a,b,output_valid,result);",
            "initial begin repeat(2) @(negedge clk); reset=0;"]
    for a in range(codes):
        for b in range(codes):
            expected = getattr(fmt, operation)(a, b)
            rows.append(f"@(negedge clk); a={bits}'d{a}; b={bits}'d{b}; input_valid=1;")
            rows.append(f"repeat({latency}) @(posedge clk); #1;")
            rows.append(f"if (output_valid !== 1'b1 || result !== {bits}'d{expected}) $fatal(1, \"pair {a},{b}\");")
    rows.extend([f'$display("PASS {codes*codes} pairs"); $finish;', "end endmodule"])
    return "\n".join(rows) + "\n"
