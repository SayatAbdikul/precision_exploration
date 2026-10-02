"""Verification helpers added in revision r3 of Experiment B2 (after independent review 2)."""
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tools.experiment_b2 import verify  # noqa: E402


def test_tie_statistics_on_a_hand_checkable_case():
    # image 0: unique maximum at the label                      -> correct under every rule
    # image 1: two-way tie that includes the label              -> strict 0, expected 1/2, optimistic 1
    # image 2: three-way tie that does not include the label    -> wrong under every rule
    # image 3: unique maximum at another class                  -> wrong under every rule
    logits = torch.tensor([[1.0, 5.0, 2.0, 0.0],
                           [3.0, 3.0, 1.0, 0.0],
                           [2.0, 2.0, 2.0, 0.0],
                           [0.0, 1.0, 4.0, 0.0]])
    rows = [{"label": 1}, {"label": 0}, {"label": 3}, {"label": 0}]
    ties = verify.tie_statistics(logits, rows)
    assert ties["images_with_tied_top1"] == 2
    assert ties["images_with_tied_top1_involving_the_label"] == 1
    assert ties["largest_tie"] == 3
    assert ties["top1_percent_strict"] == 25.0
    assert ties["top1_percent_expected"] == 37.5
    assert ties["top1_percent_optimistic"] == 50.0
    # Lowest index wins: image 1 goes to class 0 (its label), image 2 to class 0 (not its label).
    assert ties["top1_percent_lowest_index"] == 50.0
    assert "per_image" not in ties
    detail = verify.tie_statistics(logits, rows, per_image=True)["per_image"]
    assert detail == {"tie_size": [1, 2, 3, 1], "label_among_maxima": [1, 1, 0, 0],
                      "label_is_lowest_index_maximum": [1, 1, 0, 0]}
    # A two-way tie in which the label is the higher index loses under the lowest-index rule.
    assert verify.tie_statistics(logits[[1]], [{"label": 1}])["top1_percent_lowest_index"] == 0.0
    # Without ties the three rules agree.
    plain = verify.tie_statistics(logits[[0, 3]], [rows[0], rows[3]])
    assert plain["images_with_tied_top1"] == 0
    assert plain["top1_percent_strict"] == plain["top1_percent_expected"] == plain["top1_percent_optimistic"] == 50.0


def test_verification_records_are_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    path = tmp_path / "record.json"
    assert verify.target(path, "") == path
    assert verify.target(path, "r3") == tmp_path / "record--r3.json"
    path.write_text("{}")
    with pytest.raises(SystemExit):
        verify.target(path, "")
    assert verify.target(path, "r3") == tmp_path / "record--r3.json"  # a new tag gives a new file
    (tmp_path / "record--r3.json").write_text("{}")
    with pytest.raises(SystemExit):
        verify.target(path, "r3")


def test_layout_helper_recognises_a_clamp_returned_in_the_other_memory_layout():
    expected = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5) % 256
    # What a kernel does that writes NHWC memory into an NCHW tensor: same codes, permuted positions.
    wrong = expected.permute(0, 2, 3, 1).contiguous().reshape(2, 3, 4, 5)
    assert not torch.equal(wrong, expected)
    assert verify._layout(wrong, expected) == {"code_histogram_equal_to_clamp": True,
                                               "equals_clamp_read_in_other_layout": True}
    shifted = (expected + 1) % 256
    assert verify._layout(shifted, expected)["equals_clamp_read_in_other_layout"] is False


def test_kernel_audit_covers_activation_add_mul_and_pool_operators():
    from torch.ao.quantization import get_default_qconfig_mapping
    from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.hs = nn.Conv2d(3, 4, 3, padding=1), nn.Hardswish()
            self.pool, self.fc, self.gate = nn.AdaptiveAvgPool2d(1), nn.Conv2d(4, 4, 1), nn.Hardsigmoid()
            self.b = nn.Conv2d(4, 4, 1)

        def forward(self, x):
            y = self.hs(self.a(x))
            y = y * self.gate(self.fc(self.pool(y)))
            return y + self.b(y)

    torch.manual_seed(7)
    model, x = Net().eval(), torch.randn(4, 3, 8, 8)
    torch.backends.quantized.engine = "x86"
    prepared = prepare_fx(model, get_default_qconfig_mapping("x86"), example_inputs=(x,))
    with torch.inference_mode():
        prepared(x)
    rows = verify.kernel_audit(convert_fx(prepared), x)
    kinds = [row["kind"] for row in rows]
    assert sorted(set(kinds)) == ["add", "conv", "hardsigmoid", "hardswish", "mul", "pool"]
    assert kinds.count("conv") == 3
    # The x86 kernels of the shipped default agree with the FP32 evaluation of their own operands.
    assert all(row["max_code_error"] <= 1 for row in rows)
    assert all("distinct_output_codes" in row for row in rows if row["kind"] == "conv")
