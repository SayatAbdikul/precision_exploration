#include <cmath>
#include <cfenv>
#include <omp.h>
#define PE_DEVICE inline
#define PE_FMA std::fma
#define PE_LDEXP std::ldexp
#include "kernel.h"
extern "C" int bridge_conv(const int16_t* x,const int16_t* w,double* y,Shape g,int shift,int mode) {
    if(std::fegetround()!=FE_TONEAREST || !x || !w || !y || (mode!=0 && mode!=1)) return 1;
    omp_set_num_threads(4);
    int64_t count=int64_t(g.n)*g.co*g.oh*g.ow;
    #pragma omp parallel for schedule(static)
    for(int64_t i=0;i<count;++i) y[i]=reduce_one(x,w,g,i,shift,mode);
    return 0;
}
