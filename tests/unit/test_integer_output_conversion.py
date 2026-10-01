from fractions import Fraction
import subprocess

import pytest

from tools.hardware.integer_output_conversion import (
    OutputConfig, STAGE, oracle_code, run_integrated, run_stage,
)


def _code(value, bits):
    return value & ((1 << bits) - 1)


def test_exact_power_of_two_identity_and_rejection():
    assert OutputConfig.from_ratio(32, 4, Fraction(1, 2)).scale_exponent == -1
    assert OutputConfig.from_ratio(64, 8, 4).scale_exponent == 2
    for scale in (Fraction(3, 4), Fraction(5, 1), 0, -1, True):
        with pytest.raises(ValueError):
            OutputConfig.from_ratio(32, 4, scale)
    for values in ((16, 4, 0), (32, 7, 0), (32, 4, -33), (64, 8, 9)):
        with pytest.raises(ValueError):
            OutputConfig(*values)
    with pytest.raises(ValueError):
        OutputConfig(32, 4, 0, relu=1)


def test_oracle_boundaries_and_negative_ties():
    config = OutputConfig(32, 4, -1)
    expected = {1: 0, -1: 0, 3: 2, -3: -2, 5: 2, -5: -2,
                15: 7, -15: -8, -19: -8}
    for raw, output in expected.items():
        assert oracle_code(_code(raw, 32), config) == _code(output, 4)
    relu = OutputConfig(32, 4, -1, True)
    assert oracle_code(_code(-19, 32), relu) == 0


@pytest.mark.parametrize("config", [OutputConfig(32, 4, -32), OutputConfig(32, 5, -1),
                                      OutputConfig(64, 6, 0, True), OutputConfig(64, 8, 8)])
def test_stage_oracle_simulation_with_valid_bubbles(tmp_path, config):
    result = run_stage(config, tmp_path)
    assert result["status"] == "pass"
    assert result["result_checks"] > 256
    assert result["simulation_stdout"].startswith("PASS stage")


def test_integrated_model_c_bias_then_conversion(tmp_path):
    first = run_integrated(4, 4, OutputConfig(32, 4, -1), tmp_path)
    second = run_integrated(8, 8, OutputConfig(64, 8, -2, True), tmp_path)
    assert first["status"] == second["status"] == "pass"
    assert first["result_checks"] >= 293
    assert second["result_checks"] > 4000


def test_invalid_rtl_parameters_fail_elaboration():
    for parameters in (("ACC_BITS", 16), ("OUT_BITS", 7), ("SCALE_EXP", -33), ("RELU", 2)):
        result = subprocess.run(["iverilog", "-g2012", "-tnull", "-s", "integer_output_convert",
                                 "-P", f"integer_output_convert.{parameters[0]}={parameters[1]}",
                                 str(STAGE)], capture_output=True, text=True)
        assert result.returncode != 0
        assert "INVALID_INTEGER_OUTPUT_CONVERSION_PARAMETERS" in result.stderr
