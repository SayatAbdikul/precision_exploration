module smoke_arithmetic #(
  parameter integer WIDTH = 8
) (
  input  wire [WIDTH-1:0] a,
  input  wire [WIDTH-1:0] b,
  input  wire             select_mul,
  output wire [2*WIDTH-1:0] result
);
  wire [2*WIDTH-1:0] product = a * b;
  wire [WIDTH:0] sum = {1'b0, a} + {1'b0, b};
  assign result = select_mul ? product : {{(WIDTH-1){1'b0}}, sum};
endmodule

module registered_smoke_arithmetic #(
  parameter integer WIDTH = 8
) (
  input  wire               clk,
  input  wire               reset,
  input  wire               input_valid,
  input  wire [WIDTH-1:0]   a,
  input  wire [WIDTH-1:0]   b,
  input  wire               select_mul,
  output reg                output_valid,
  output reg  [2*WIDTH-1:0] result
);
  reg  [WIDTH-1:0] a_q;
  reg  [WIDTH-1:0] b_q;
  reg              select_mul_q;
  reg              valid_q;
  wire [2*WIDTH-1:0] combinational_result;

  smoke_arithmetic #(.WIDTH(WIDTH)) dut (
    .a(a_q),
    .b(b_q),
    .select_mul(select_mul_q),
    .result(combinational_result)
  );

  always @(posedge clk) begin
    if (reset) begin
      a_q <= {WIDTH{1'b0}};
      b_q <= {WIDTH{1'b0}};
      select_mul_q <= 1'b0;
      valid_q <= 1'b0;
      output_valid <= 1'b0;
      result <= {2*WIDTH{1'b0}};
    end else begin
      a_q <= a;
      b_q <= b;
      select_mul_q <= select_mul;
      valid_q <= input_valid;
      output_valid <= valid_q;
      if (valid_q)
        result <= combinational_result;
    end
  end
endmodule
