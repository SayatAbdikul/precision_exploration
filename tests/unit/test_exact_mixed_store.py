from fractions import Fraction

import pytest

from public.inference.operators.dispatch import Operators
from public.inference.tensor import Encoding, QUANTIZATION_OBSERVER
from tools.exact_execution_v2.mixed_store import mixed_integer_store


@pytest.mark.parametrize('bits',[32,64])
@pytest.mark.parametrize('output_bits',[4,8])
@pytest.mark.parametrize('activation',['identity','relu','relu6'])
def test_mixed_scale_channels_match_every_original_code_and_observer_value(bits,output_bits,activation):
    signed=[-(1<<(bits-1)),-255,-31,-1,0,1,31,255,(1<<(bits-1))-1]
    states=tuple(v&((1<<bits)-1) for v in signed*6)
    args=(states,(2,3,3,3),f'int{bits}_accumulator',
          (Fraction(1,10),Fraction(1,3),Fraction(1,10**45)),
          ['0.1','-0.3','0'],activation,Encoding(f'int{output_bits}',('0.2',)))
    observed=[]
    token=QUANTIZATION_OBSERVER.set(lambda values,tensor:observed.append((tuple(values),tensor)))
    fallback_calls=[]
    def fallback(*args):
        fallback_calls.append(args)
        return Operators()._store(*args)
    try:
        expected=Operators()._store(*args)
        actual=mixed_integer_store(*args,fallback=fallback)
    finally:
        QUANTIZATION_OBSERVER.reset(token)
    assert actual==expected
    assert len(observed)==2 and observed[0]==observed[1]
    assert len(fallback_calls)==2  # 1/3 and 1e-45 remain outside the original bound.
    assert all(call[1]==(2,1,3,3) for call in fallback_calls)


def test_wholly_supported_or_wholly_unsupported_domains_keep_existing_dispatch():
    def forbidden(*args):
        raise AssertionError('fallback must not be invoked while declining dispatch')
    for scales in ((Fraction(1),Fraction(2)),(Fraction(1,3),Fraction(1,10**45))):
        assert mixed_integer_store((0,1),(1,2),'int32_accumulator',scales,None,'identity',Encoding('int4'),fallback=forbidden) is None


def test_mixed_stores_keep_saturating_bias_and_channel_order():
    states=tuple(((-1)**i*(2**63-1))&((1<<64)-1) for i in range(12))
    args=(states,(2,3,2),'int64_accumulator',(Fraction(1),Fraction(1,10**45),Fraction(3,10)),
          [str(2**100),str(-(2**100)),'1.1'],'identity',Encoding('int8'))
    assert mixed_integer_store(*args,fallback=Operators()._store)==Operators()._store(*args)
