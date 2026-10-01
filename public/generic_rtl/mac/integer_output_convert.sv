// Exact, static power-of-two integer accumulator-to-output conversion.
// Supported: INT32/INT64 input, INT4/5/6/8 output, RNE, saturation,
// optional ReLU, and SCALE_EXP in [-ACC_BITS, OUT_BITS].
// Arithmetic meaning: Q_OUT(max(0, ACC * 2^SCALE_EXP)) for RELU=1;
// otherwise Q_OUT(ACC * 2^SCALE_EXP). The accumulator already includes bias.
module integer_output_convert #(
    parameter integer ACC_BITS = 32,
    parameter integer OUT_BITS = 4,
    parameter integer SCALE_EXP = 0,
    parameter integer RELU = 0
) (
    input wire clk,
    input wire reset,
    input wire input_valid,
    input wire signed [ACC_BITS-1:0] accumulator_code,
    output reg output_valid,
    output reg signed [OUT_BITS-1:0] output_code
);
    generate if (!((ACC_BITS == 32 || ACC_BITS == 64) &&
                   (OUT_BITS == 4 || OUT_BITS == 5 || OUT_BITS == 6 || OUT_BITS == 8) &&
                   (SCALE_EXP >= -ACC_BITS && SCALE_EXP <= OUT_BITS) &&
                   (RELU == 0 || RELU == 1))) begin: invalid_parameters
        INVALID_INTEGER_OUTPUT_CONVERSION_PARAMETERS illegal_configuration();
    end endgenerate

    localparam [ACC_BITS:0] ONE = {{ACC_BITS{1'b0}}, 1'b1};
    localparam [ACC_BITS:0] NEGATIVE_LIMIT = ONE << (OUT_BITS-1);
    localparam [ACC_BITS:0] POSITIVE_LIMIT = NEGATIVE_LIMIT - 1'b1;
    wire negative = accumulator_code[ACC_BITS-1];
    wire signed [ACC_BITS:0] extended_code =
        {accumulator_code[ACC_BITS-1], accumulator_code};
    wire [ACC_BITS:0] magnitude = negative ? -extended_code : extended_code;
    wire [ACC_BITS:0] limit = negative ? NEGATIVE_LIMIT : POSITIVE_LIMIT;
    wire [ACC_BITS:0] rounded_magnitude;

    generate if (SCALE_EXP < 0) begin: round_right
        localparam integer SHIFT = -SCALE_EXP;
        wire [ACC_BITS:0] quotient = magnitude >> SHIFT;
        wire [ACC_BITS:0] remainder = magnitude & ((ONE << SHIFT) - 1'b1);
        wire [ACC_BITS:0] halfway = ONE << (SHIFT-1);
        wire round_up = (remainder > halfway) ||
                        ((remainder == halfway) && quotient[0]);
        assign rounded_magnitude = quotient + round_up;
    end else begin: shift_left
        // The comparison precedes the left shift so even INT64_MIN cannot
        // overflow an intermediate. A value above the output limit saturates.
        assign rounded_magnitude = magnitude > (limit >> SCALE_EXP) ?
                                   limit + 1'b1 : magnitude << SCALE_EXP;
    end endgenerate

    wire [ACC_BITS:0] bounded_magnitude =
        rounded_magnitude > limit ? limit : rounded_magnitude;
    wire signed [OUT_BITS:0] signed_magnitude =
        $signed({1'b0, bounded_magnitude[OUT_BITS-1:0]});
    wire signed [OUT_BITS:0] signed_value =
        (RELU && negative) ? {(OUT_BITS+1){1'b0}} :
        negative ? -signed_magnitude : signed_magnitude;

    always @(posedge clk) begin
        if (reset) begin
            output_valid <= 1'b0;
            output_code <= {OUT_BITS{1'b0}};
        end else begin
            output_valid <= input_valid;
            if (input_valid) output_code <= signed_value[OUT_BITS-1:0];
        end
    end
endmodule
