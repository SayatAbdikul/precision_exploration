from public.analysis.phase3.posit_dfl_cases import selected_vectors
from public.inference.reference.arithmetic import format_named
from public.inference.tensor import Encoding


def test_posit_cases_exclude_nar_preserve_endpoints_and_reproduce_seed():
    for name in ("posit4_es0", "posit6_es1", "posit8_es1"):
        x = Encoding(name)
        fmt = format_named(name)
        cases = selected_vectors(x, 16)
        assert cases == selected_vectors(x, 16)
        assert len(set(cases)) == len(cases)
        finite = [c for c in range(1 << fmt.bits) if fmt.decode(c).is_finite()]
        assert all(len(row) == 16 and set(row) <= set(finite) for row in cases)
        assert all((c,)*16 in cases for c in finite)
        assert all((0,)+(c,)*15 in cases and (c,)*15+(0,) in cases for c in finite)
