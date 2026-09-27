"""Replay the frozen activation oracle's cache history across checkpoint resumes.

The historical oracle emits pre-store events on cache misses. Caches affect its
diagnostic event counts even though output codes are unchanged. Record ordered
oracle calls, so a fresh process can reconstruct exactly the previous cache
state without rerunning completed images. Each comparison pass starts cold.
"""
from fractions import Fraction

from public.inference.reference import operators as ref
from public.inference.tensor import Encoding, QUANTIZATION_OBSERVER


def pack_encoding(e):
    return {'format':e.format, 'scales':[[s.numerator,s.denominator] for s in e.scales],
            'axis':e.axis,'block_size':e.block_size}


def unpack_encoding(e):
    return Encoding(e['format'],tuple(Fraction(n,d) for n,d in e['scales']),e['axis'],e['block_size'])


class ActivationTrace:
    def __enter__(self):
        self.original=ref._activation_code
        self.original.cache_clear()
        self.calls=[]
        def capture(code,input_encoding,function,output,accumulator):
            self.calls.append([code,pack_encoding(input_encoding),function,pack_encoding(output),accumulator])
            return self.original(code,input_encoding,function,output,accumulator)
        ref._activation_code=capture
        return self

    def __exit__(self,*_):
        ref._activation_code=self.original

    def replay(self,calls):
        token=QUANTIZATION_OBSERVER.set(None)
        try:
            for code,source,function,output,accumulator in calls:
                self.original(code,unpack_encoding(source),function,unpack_encoding(output),accumulator)
        finally:
            QUANTIZATION_OBSERVER.reset(token)
