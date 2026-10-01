"""Non-inference ledger regression tests; numerical checks run under its budget."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import pytest
from tools.scaled_bridge_v1 import controller as c
from tools.scaled_bridge_v1.common import immutable

def test_open_reservation_is_charged_and_cannot_overbook():
    with tempfile.TemporaryDirectory() as tmp,patch.object(c,'BASE',Path(tmp)):
        p=Path(tmp)/'budget/attempts/one.json'
        immutable(p,{'ceiling':14400.,'reserved_seconds':14000.})
        assert c.ledger()['remaining_seconds']==400
        with pytest.raises(c.BudgetStop):c.dispatch(['check'],401,{})
        immutable(p.with_name('one.complete'),{'charged_seconds':25.,'status':'complete'})
        assert c.ledger()['remaining_seconds']==14375

def test_failed_attempt_is_not_forgotten():
    with tempfile.TemporaryDirectory() as tmp,patch.object(c,'BASE',Path(tmp)):
        p=Path(tmp)/'budget/attempts/one.json'
        immutable(p,{'ceiling':14400.,'reserved_seconds':600.})
        immutable(p.with_name('one.complete'),{'charged_seconds':600.,'status':'failed'})
        assert c.ledger()['charged_seconds']==600
        with pytest.raises(ValueError):immutable(p,{'ceiling':14400.,'reserved_seconds':1.})

def test_resume_cost_only_charges_unfinished_images_and_has_margin():
    row={'timing':{'execution':4.},'preprocess_seconds':1.,'setup_seconds':20.}
    with patch.object(c,'records',return_value=[row]*8):
        assert c.panel_reservation('case','wide','cuda',8)==0
        assert c.panel_reservation('case','wide','cuda',32)>=1.5*24*5+60

def test_failed_dispatch_keeps_full_reservation():
    with tempfile.TemporaryDirectory(dir=c.ROOT/'build') as tmp,patch.object(c,'BASE',Path(tmp)),patch.object(c,'sources',return_value={}):
        with patch.object(c.subprocess,'Popen') as launch:
            launch.return_value.wait.return_value=1
            with pytest.raises(RuntimeError):c.dispatch(['check'],90,{'cpu':30,'cuda':31})
        assert c.ledger()['charged_seconds']==90
        assert c.ledger()['attempts'][0]['completion']['status']=='failed'

def test_first_difference_uses_fx_order_not_json_key_order():
    from tools.analysis.scaled_bridge_audit import first_divergences
    a={'layers':{'add':{'codes':'a'},'conv1':{'codes':'a'},'x':{'codes':'x'}}}
    b={'layers':{'add':{'codes':'b'},'conv1':{'codes':'b'},'x':{'codes':'x'}}}
    assert first_divergences([a],[b],['x','conv1','add'])=={'conv1':1}
