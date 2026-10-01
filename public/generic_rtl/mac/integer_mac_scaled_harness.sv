// Neutral launch and capture wrapper around MAC-plus-conversion DUT.
module integer_mac_scaled_harness #(
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
    output reg output_valid,
    output reg signed [OUT_BITS-1:0] result
);
    reg valid_launch, start_launch, end_launch;
    reg signed [W_BITS-1:0] weight_launch;
    reg signed [A_BITS-1:0] activation_launch;
    reg signed [ACC_BITS-1:0] bias_launch;
    wire valid_dut;
    wire signed [OUT_BITS-1:0] result_dut;

    integer_mac_scaled #(.W_BITS(W_BITS), .A_BITS(A_BITS),
        .ACC_BITS(ACC_BITS), .OUT_BITS(OUT_BITS),
        .SCALE_EXP(SCALE_EXP), .RELU(RELU)) dut (
        .clk(clk), .reset(reset), .input_valid(valid_launch),
        .dot_start(start_launch), .dot_end(end_launch),
        .weight(weight_launch), .activation(activation_launch),
        .bias(bias_launch), .output_valid(valid_dut), .result(result_dut));

    always @(posedge clk) begin
        if (reset) begin
            valid_launch <= 1'b0;
            start_launch <= 1'b0;
            end_launch <= 1'b0;
            weight_launch <= 0;
            activation_launch <= 0;
            bias_launch <= 0;
            output_valid <= 1'b0;
            result <= 0;
        end else begin
            valid_launch <= input_valid;
            start_launch <= dot_start;
            end_launch <= dot_end;
            weight_launch <= weight;
            activation_launch <= activation;
            bias_launch <= bias;
            output_valid <= valid_dut;
            result <= result_dut;
        end
    end
endmodule
