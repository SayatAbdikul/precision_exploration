// Public generic integer Model C dot-product baseline, version 1.
// Operand codes are signed two's-complement integers at scale=1. The
// accumulator is the accepted signed INT32/INT64 saturating code domain.
// External scale application, activation and output requantization are absent.
module integer_mac #(
    parameter integer W_BITS = 4,
    parameter integer A_BITS = 4,
    parameter integer ACC_BITS = 32
) (
    input wire clk,
    input wire reset,
    input wire input_valid,
    input wire dot_start,
    input wire dot_end,
    input wire signed [W_BITS-1:0] weight,
    input wire signed [A_BITS-1:0] activation,
    input wire signed [ACC_BITS-1:0] bias,
    output reg output_valid,
    output reg signed [ACC_BITS-1:0] result
);
    localparam integer PRODUCT_BITS = W_BITS + A_BITS;
    localparam signed [ACC_BITS-1:0] ACC_MAX =
        {1'b0, {(ACC_BITS-1){1'b1}}};
    localparam signed [ACC_BITS-1:0] ACC_MIN =
        {1'b1, {(ACC_BITS-1){1'b0}}};
    localparam signed [ACC_BITS:0] ACC_MAX_EXT = {1'b0, ACC_MAX};
    localparam signed [ACC_BITS:0] ACC_MIN_EXT = {1'b1, ACC_MIN};

    reg signed [ACC_BITS-1:0] acc_q;
    wire signed [PRODUCT_BITS-1:0] product = weight * activation;
    wire signed [ACC_BITS:0] product_ext =
        {{(ACC_BITS+1-PRODUCT_BITS){product[PRODUCT_BITS-1]}}, product};
    wire signed [ACC_BITS:0] base_ext =
        dot_start ? {(ACC_BITS+1){1'b0}} : {acc_q[ACC_BITS-1], acc_q};
    wire signed [ACC_BITS:0] mac_sum = base_ext + product_ext;

    function automatic signed [ACC_BITS-1:0] saturate;
        input signed [ACC_BITS:0] value;
        begin
            if (value > ACC_MAX_EXT) saturate = ACC_MAX;
            else if (value < ACC_MIN_EXT) saturate = ACC_MIN;
            else saturate = value[ACC_BITS-1:0];
        end
    endfunction

    wire signed [ACC_BITS-1:0] mac_rounded = saturate(mac_sum);
    wire signed [ACC_BITS:0] bias_sum =
        {mac_rounded[ACC_BITS-1], mac_rounded} + {bias[ACC_BITS-1], bias};
    wire signed [ACC_BITS-1:0] biased_result = saturate(bias_sum);

    always @(posedge clk) begin
        if (reset) begin
            acc_q <= {ACC_BITS{1'b0}};
            result <= {ACC_BITS{1'b0}};
            output_valid <= 1'b0;
        end else begin
            output_valid <= input_valid && dot_end;
            if (input_valid) begin
                acc_q <= mac_rounded;
                if (dot_end) result <= biased_result;
            end
        end
    end
endmodule
