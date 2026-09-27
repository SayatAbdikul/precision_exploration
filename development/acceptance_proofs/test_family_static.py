"""Static classifier composition remains separate from native acceptance."""
import pytest

from development.acceptance_proofs.family_static import verify_static
from tools.phase3.common import ROOT, read
from tools.phase3.preparation_inventory import preparation_records


def prepared(key):
    row = preparation_records()[key]
    return read((ROOT / row['configuration']['path']).parent / 'prepared.json')


@pytest.mark.parametrize('key', ('resnet18/q1_6', 'resnet18/fp4_e2m1', 'resnet18/posit4_es0'))
def test_three_families_recompute_complete_classifier_arithmetic(key):
    result = verify_static(prepared(key))
    assert result['status'] == 'static_graph_arithmetic_reproduced_native_pilot_required'
    assert result['mac_nodes'] + result['nonmac_nodes'] == 49


@pytest.mark.parametrize('key', ('resnet18/fp8_e4m3fn', 'yolov8n/q1_6'))
def test_sensitive_or_detector_graph_cannot_be_promoted(key):
    with pytest.raises(ValueError, match='complete local classifier proof|remains pending'):
        verify_static(prepared(key))
