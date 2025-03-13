// #include <math.h>
#include <iostream>

__global__ void shift_add_forward(const float *input, float *out, 
        const int *idxes, 
        const int *shift_pads, int extra_pad,
        int nk, int nthreads,
        int b,int c_in, int hin, int win,
        int c_out, int hout, int wout, bool isH)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nthreads; i += gridDim.x * blockDim.x)
    {
        int pw = i % wout;
        // all layer share same valuable span in height for slide window along horizon
        int ph = (i / wout) % hout;
        int pc = (i / wout / hout) % c_out;
        int pb = i / wout / hout / c_out;
        float s = 0.0;
        for(int ii=0;ii<nk;ii++){
            int pad = shift_pads[ii];
	    // share same arrange order
            int idx = idxes[pc * nk +ii];//pc * nk +
            if(isH>0){
                int ww = pw - pad;
                if (0 <= ww && ww <= win-1){
                    int hh = ph + extra_pad;
                    s += input[pb*(c_in*win*hin)+idx*(win*hin)+hh*win+ww];
                }
            }else{
                int hh = ph - pad;
                if (0 <= hh && hh <= hin-1){
                    int ww = pw + extra_pad;
                    s += input[pb*(c_in*win*hin)+idx*(win*hin)+hh*win+ww];
                }
            }
        }
        out[pb*(wout*hout*c_out)+pc*(wout*hout)+ph*wout+pw] = s;
    }
}


__global__ void shift_add_backward(float *ginput, float *gout, 
        const int *idxes, 
        const int *shift_pads, int extra_pad,
        int nk, int nthreads,
        int b,int c_in, int hin, int win,
        int c_out, int hout, int wout, bool isH)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nthreads; i += gridDim.x * blockDim.x)
    {
        int pw = i % wout;
        // all layer share same valuable span in height for slide window along horizon
        int ph = (i / wout) % hout;
        int pc = (i / wout / hout) % c_out;
        int pb = i / wout / hout / c_out;
 
        float gs = gout[pb*(wout*hout*c_out)+pc*(wout*hout)+ph*wout+pw];
        for(int ii=0;ii<nk;ii++){
            int pad = shift_pads[ii];
	    // share same arrange order
            int idx = idxes[pc * nk +ii];//pc * nk +
            if(isH>0){
                int ww = pw - pad;
                if (0 <= ww && ww <= win-1){
                    int hh = ph + extra_pad;
                    ginput[pb*(c_in*win*hin)+idx*(win*hin)+hh*win+ww]=gs;
                }
            }else{
                int hh = ph - pad;
                if (0 <= hh && hh <= hin-1){
                    int ww = pw + extra_pad;
                    ginput[pb*(c_in*win*hin)+idx*(win*hin)+hh*win+ww]=gs;
                }
            }
        }
    }
}
 

// void launch_shift_add_v0(float *out, const float *input,
void launch_shift_add_forward_v0(float *out, const float *input,
                                const int *idxes, const int *shift_pads, int extra_pad,
                                int b, int c_in, int hin, int win,
                                int c_out, int hout, int wout, bool isH)
{
    int nthreads = wout * hout * c_out * b;
    int nk = ceil(c_in / c_out);
    dim3 grid(ceil(nthreads / 512.0 / 8.0));
    dim3 block(512);
    // dim3 grid(1);
    // dim3 block(16);
    // std::cout << "------------------" << ceil(nthreads / 8 / 512) << "------------------" << nthreads << "------------------" << std::endl;
    // for(int ii=0;ii<threadperline;ii++){
    //   std::cout<<shift_tuple[ii*2]<<",y="<<shift_tuple[ii*2+1]<<std::endl;
    // }
    shift_add_forward<<<grid, block>>>(input, out, idxes, shift_pads, extra_pad,
                                        nk, nthreads, b, c_in, hin, win, c_out, hout, wout, isH);
}

void launch_shift_add_backward_v0(float *gout, float *ginput,
                                const int *idxes, const int *shift_pads, int extra_pad,
                                int b, int c_in, int hin, int win,
                                int c_out, int hout, int wout, bool isH)
{
    int nthreads = wout * hout * c_out * b;
    int nk = ceil(c_in / c_out);
    int nt = ceil(nthreads / 512.0 / 8.0);
    nt = nt>3500? 2048 : 1024;
    dim3 grid(nt);
    dim3 block(512);
    // dim3 grid(1);
    // dim3 block(16);
    // std::cout << "------------------" << ceil(nthreads / 8 / 512) << "------------------" << nthreads << "------------------" << std::endl;
    // for(int ii=0;ii<threadperline;ii++){
    //   std::cout<<shift_tuple[ii*2]<<",y="<<shift_tuple[ii*2+1]<<std::endl;
    // }
    shift_add_backward<<<grid, block>>>(ginput, gout, idxes, shift_pads, extra_pad,
                                        nk, nthreads, b, c_in, hin, win, c_out, hout, wout, isH);
}


