"""NCHW to K-block convolution: explicit scale ownership and requantization."""
from fractions import Fraction

from public.inference.tensor import Tensor, Encoding, SharedEncoding, indices, offset
from .arithmetic import real, binary, model_c, format_named
from .operators import pair, _accumulator, post_activation


def block_conv2d(inputs, weights, *, kernel_size, patch_format, accumulator, output,
                 bias=None, stride=1, padding=0, dilation=1, groups=1, activation="identity", native=None):
    if len(inputs.shape) != 4 or len(weights.shape) != 2:
        raise ValueError("block convolution requires NCHW activations and O x K weights")
    n,ci,h,w = inputs.shape
    co,k = weights.shape
    kh,kw = pair(kernel_size)
    sh,sw = pair(stride); ph,pw = pair(padding); dh,dw = pair(dilation)
    if (type(groups) is not int or groups <= 0 or ci % groups or co % groups or k != ci//groups*kh*kw
            or min(kh,kw,sh,sw,dh,dw) <= 0 or min(ph,pw) < 0):
        raise ValueError("invalid K-block convolution geometry")
    if weights.encoding.block_size and weights.encoding.axis != 1:
        raise ValueError("convolution weight blocks must own flattened K")
    if bias is not None and len(bias) != co:
        raise ValueError("bias count mismatch")
    oh,ow = (h+2*ph-dh*(kh-1)-1)//sh+1,(w+2*pw-dw*(kw-1)-1)//sw+1
    if min(oh,ow) <= 0:
        raise ValueError("empty block convolution output")
    acc = _accumulator(accumulator)
    if acc.family not in {"float","fixed_point"}:
        raise ValueError("cross-block reduction requires an explicit linear scaled-value accumulator")
    values = [None]*(n*co*oh*ow)
    for group in range(groups):
        patch_values = []
        for b,y,x in indices((n,oh,ow)):
            for c,kr,kc in indices((ci//groups,kh,kw)):
                iy,ix = y*sh-ph+kr*dh,x*sw-pw+kc*dw
                patch_values.append(inputs.value((b,group*(ci//groups)+c,iy,ix)) if 0 <= iy < h and 0 <= ix < w else Fraction(0))
        if format_named(patch_format).manifest["scaling"]["mode"] == "intrinsic_shared":
            patches = Tensor.quantize(patch_values,(n*oh*ow,k),SharedEncoding(patch_format,axis=1))
            patch_values = patches.values()
        elif patch_format != inputs.encoding.format:
            raise ValueError("ordinary patches must preserve the input encoding")
        # Ordinary activations retain their already stored values, including
        # mathematical padding zero for formats without an encoded zero.
        first,last = group*(co//groups),(group+1)*(co//groups)
        encoding = weights.encoding
        if encoding.axis is not None:
            per_row = len(encoding.scales)//co if encoding.block_size else 1
            encoding = Encoding(encoding.format,encoding.scales[first*per_row:last*per_row],encoding.axis,encoding.block_size)
        selected = Tensor((co//groups,k),weights.codes[first*k:last*k],encoding)
        if native is None:
            states = [model_c(((patch_values[row*k+i],selected.value((c,i))) for i in range(k)),acc)
                      for row in range(n*oh*ow) for c in range(co//groups)]
        else:
            # Scales are decoded before products. The accumulator remains live
            # across every block and the final partial block; no block subtotal
            # or intermediate output rounding changes sequential Model C.
            states = native.flex_gemm(patch_values,selected.values(),batch=n*oh*ow,channels=co//groups,k=k,accumulator=accumulator)
        for row,(b,y,x) in enumerate(indices((n,oh,ow))):
            for channel in range(co//groups):
                value = acc.decode(states[row*(co//groups)+channel])
                if bias is not None:
                    value = acc.rounded(binary(value,acc.rounded(real(bias[first+channel])),"add"))
                values[offset((n,co,oh,ow),(b,first+channel,y,x))] = post_activation(value,activation)
    return Tensor.quantize(values,(n,co,oh,ow),output)
