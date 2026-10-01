// Model C raw integer MAC plus one exact static power-of-two output stage.
// Bias is included in the MAC before conversion/activation/output rounding.
module integer_mac_scaled #(
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer ACC_BITS = 32,
    parameter integer OUT_BITS = 4,
    parameter integer SCALE_EXP = 0,
    parameter integer RELU = 0
) (
    input wire clk, reset, input_valid, dot_start, dot_end,
    input wire signed [W_BITS-1:0] weight,
    input wire signed [A_BITS-1:0] activation,
    input wire signed [ACC_BITS-1:0] bias,
    output wire output_valid,
    output wire signed [OUT_BITS-1:0] result
);
    wire mac_valid;
    wire signed [ACC_BITS-1:0] mac_result;
    integer_mac #(.W_BITS(W_BITS), .A_BITS(A_BITS), .ACC_BITS(ACC_BITS)) mac (
        .clk(clk), .reset(reset), .input_valid(input_valid),
        .dot_start(dot_start), .dot_end(dot_end), .weight(weight),
        .activation(activation), .bias(bias),
        .output_valid(mac_valid), .result(mac_result));
    integer_output_convert #(.ACC_BITS(ACC_BITS), .OUT_BITS(OUT_BITS),
        .SCALE_EXP(SCALE_EXP), .RELU(RELU)) conversion (
        .clk(clk), .reset(reset), .input_valid(mac_valid),
        .accumulator_code(mac_result), .output_valid(output_valid),
        .output_code(result));
endmodule
