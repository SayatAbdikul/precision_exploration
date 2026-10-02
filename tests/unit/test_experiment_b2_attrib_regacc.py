"""Lane Q4 r7: v2.1 accumulator correction uses each network's own certificate (CPU, synthetic + sealed files)."""
import pytest

from tools.experiment_b2_attrib import regacc


def _meta(layers, net_default, net_arm):
    return {"accumulator": {"layers": layers, "network_max_bits_default": net_default, "network_max_bits_arm": net_arm}}


def test_matched_accumulator_takes_unchanged_layers_from_the_given_certificate():
    cert = {"a": 27, "b": 20, "c": 19}
    meta = _meta({"b": {"bits_default": 20, "bits_arm": 21, "bits_arm_presubtracted_operand": 22, "zero_point": -15}},
                 27, 27)
    acc = regacc.accumulator_matched(meta, cert)
    assert acc["network_max_bits_arm_check"] == 27          # widest unchanged layer "a"
    assert acc["network_max_bits_presubtracted"] == 27
    assert acc["certificate_matches_layer_defaults"] is True
    assert acc["layers_with_increase"] == 1 and acc["presubtracted_increase_layers"] == ["b"]
    assert acc["zero_points"] == [-15] and acc["network_max_bits_certificate_default"] == 27
    # a changed layer that becomes the widest sets the network maximum
    meta = _meta({"a": {"bits_default": 27, "bits_arm": 28, "bits_arm_presubtracted_operand": 29}}, 27, 28)
    acc = regacc.accumulator_matched(meta, cert)
    assert acc["network_max_bits_arm_check"] == 28 and acc["network_max_bits_presubtracted"] == 29
    assert acc["presubtracted_widest_layers"] == ["a"]


def test_matched_accumulator_flags_a_foreign_certificate():
    meta = _meta({"conv1": {"bits_default": 24, "bits_arm": 24}}, 27, 27)
    acc = regacc.accumulator_matched(meta, {"features_0_0": 19, "classifier_3": 24})
    assert acc["certificate_matches_layer_defaults"] is False
    assert acc["layers_missing_from_certificate"] == ["conv1"]


@pytest.mark.parametrize("model,fmt,expected", [("mobilenet_v3_large", "int8", 24), ("mobilenet_v3_large", "int6", 20),
                                                ("mobilenet_v2", "int8", 25), ("resnet18", "int8", 27)])
def test_certificate_paths_are_per_model(model, fmt, expected):
    path = regacc.certificate_path(model, fmt)
    if not path.exists():
        pytest.skip(f"sealed certificate missing: {path}")
    assert path.parent.name == f"{model}-{fmt}-default-b2"
    assert max(regacc.certificate_bits(model, fmt).values()) == expected
