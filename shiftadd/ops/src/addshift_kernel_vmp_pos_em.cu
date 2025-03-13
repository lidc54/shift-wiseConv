// #include <math.h>
#include <iostream>
#include <cuda_runtime.h>

__global__ void shift_add_mp_forward(const float *input, float *out, 
        const int *pad_hv, const int *idx_identit, const int *idx_out, 
        int extra_pad, int nk, int nthreads,
        int b,int c_in, int hin, int win,
        int c_out, int hout, int wout, int group_in,
        const float *w1, float *w2, float *w3)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nthreads; i += gridDim.x * blockDim.x)
    {
        //loop input
        int pw = i % win;
        int ph = (i / win) % hin;
        int pc = (i / win / hin) % c_in;
        int pb = i / win / hin / c_in;

        int oc = idx_out[pc], w_hv = 2 * group_in;
        int total=wout*hout*c_out*b;
        float x = input[pb*(c_in*win*hin)+pc*(win*hin)+ph*win+pw];
        
        for(int ii=0;ii<group_in;ii++){
            // need transform to cin*group; then concate along dim=1
            int pad_h = pad_hv[pc*w_hv+ii];
            int pad_v = pad_hv[pc*w_hv+ii+group_in];
            // transform to cout*group
            int idx_i = idx_identit[oc*group_in+ii];
            float ww1 = w1[pc*group_in+ii];
            float ww2 = w2[pc*group_in+ii];

            // horizon
            int wi_h = pw + pad_h;
            int hi_h = ph - extra_pad;
            if (0 <= wi_h && wi_h <= wout-1 && 0 <= hi_h && hi_h <= hout-1 ){
                float *y = out + 0*total + pb*(wout*hout*c_out)+oc*(wout*hout)+hi_h*wout+wi_h;
                atomicAdd(y, x+ww1);
            }
            // vertical
            int wi_v = pw - extra_pad;
            int hi_v = ph + pad_v;
            if (0 <= wi_v && wi_v <= wout-1 && 0 <= hi_v && hi_v <= hout-1 ){
                float *y = out + 1*total + pb*(wout*hout*c_out)+oc*(wout*hout)+hi_v*wout+wi_v;
                atomicAdd(y, x+ww2);
            }
            // identity
            if (pc == idx_i){
                float ww3 = w3[(pc/nk)*group_in+ii];
                int wi_i = pw - extra_pad;
                int hi_i = ph - extra_pad;
                if (0 <= wi_i && wi_i <= wout-1 && 0 <= hi_i && hi_i <= hout-1 ){
                    float *y = out + 2*total + pb*(wout*hout*c_out)+oc*(wout*hout)+hi_i*wout+wi_i;
                    atomicAdd(y, x+ww3);
                }
            }
        }
    }
}

__global__ void shift_add_mp_backward(float *ginput, float *gout, 
        const int *pad_hv, const int *idx_identit, const int *idx_out, 
        int extra_pad, int nk, int nthreads,
        int b,int c_in, int hin, int win,
        int c_out, int hout, int wout, int group_in)
        // ,float *w1, float *w2, float *w3)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < nthreads; i += gridDim.x * blockDim.x)
    {
        int pw = i % win;
        int ph = (i / win) % hin;
        int pc = (i / win / hin) % c_in;
        int pb = i / win / hin / c_in;

        int oc = idx_out[pc], w_hv = 2 * group_in;
        int total=wout*hout*c_out*b;
        float *x = ginput + pb*(c_in*win*hin)+pc*(win*hin)+ph*win+pw;

        for(int ii=0;ii<group_in;ii++){
            // need transform to cin*group; then concate along dim=1
            int pad_h = pad_hv[pc*w_hv+ii];
            int pad_v = pad_hv[pc*w_hv+ii+group_in];
            // transform to cout*group
            int idx_i = idx_identit[oc*group_in+ii];
            // float ww1 = w1[pc*group_in+ii];
            // float ww2 = w2[pc*group_in+ii];

            // horizon
            int wi_h = pw + pad_h;
            int hi_h = ph - extra_pad;
            if (0 <= wi_h && wi_h <= wout-1 && 0 <= hi_h && hi_h <= hout-1 ){
                float y = gout[0*total + pb*(wout*hout*c_out)+oc*(wout*hout)+hi_h*wout+wi_h];
                atomicAdd(x, y);//ww1*
            }
            // vertical
            int wi_v = pw - extra_pad;
            int hi_v = ph + pad_v;
            if (0 <= wi_v && wi_v <= wout-1 && 0 <= hi_v && hi_v <= hout-1 ){
                float y = gout[1*total + pb*(wout*hout*c_out)+oc*(wout*hout)+hi_v*wout+wi_v];
                atomicAdd(x, y);//ww2*
            }
            // identity
            if (pc == idx_i){
                // float ww3 = w3[oc*group_in+ii];
                int wi_i = pw - extra_pad;
                int hi_i = ph - extra_pad;
                if (0 <= wi_i && wi_i <= wout-1 && 0 <= hi_i && hi_i <= hout-1 ){
                    float y = gout[2*total + pb*(wout*hout*c_out)+oc*(wout*hout)+hi_i*wout+wi_i];
                    atomicAdd(x, y);//ww3*
                }
            }
        }

    }
}


void launch_shift_add_forward_v_mp_linear(float *out, const float *input,
                                const int *pad_hv, const int *idx_identit, const int *idx_out, 
                                int extra_pad, int b, int c_in, int hin, int win,
                                int c_out, int hout, int wout, int group_in,
                                const float *w1, float *w2, float *w3)
{
    int nthreads = win * hin * c_in * b;
    int nk = ceil(c_in / c_out);
    dim3 grid(ceil(nthreads / 512.0 / 8.0));
    dim3 block(512);
    // dim3 grid(1);
    // dim3 block(16);
    // std::cout << "------------------" << ceil(nthreads / 8 / 512) << "------------------" << nthreads << "------------------" << std::endl;
    // for(int ii=0;ii<threadperline;ii++){
    //   std::cout<<shift_tuple[ii*2]<<",y="<<shift_tuple[ii*2+1]<<std::endl;
    // }
    shift_add_mp_forward<<<grid, block>>>(input, out, pad_hv, idx_identit, idx_out, extra_pad,
                                        nk, nthreads, b, c_in, hin, win, c_out, hout, wout, group_in,
                                        w1,w2,w3);
}


void launch_shift_add_backward_v_mp_linear(float *gout, float *ginput,
                                const int *pad_hv, const int *idx_identit, const int *idx_out, 
                                int extra_pad, int b, int c_in, int hin, int win,
                                int c_out, int hout, int wout, int group_in)
                                // ,float *w1, float *w2, float *w3)
{
    int nthreads = win * hin * c_in * b;
    int nk = ceil(c_in / c_out);
    // int nt = ceil(nthreads / 512.0 / 8.0);
    // nt = nt>3500? 2048 : 1024;
    dim3 grid(ceil(nthreads / 512.0 / 8.0));
    dim3 block(512);
    // dim3 grid(1);
    // dim3 block(16);
    // std::cout << "------------------" << ceil(nthreads / 8 / 512) << "------------------" << nthreads << "------------------" << std::endl;
    // for(int ii=0;ii<threadperline;ii++){
    //   std::cout<<shift_tuple[ii*2]<<",y="<<shift_tuple[ii*2+1]<<std::endl;
    // }
    shift_add_mp_backward<<<grid, block>>>(ginput, gout, pad_hv, idx_identit, idx_out, extra_pad,
                                        nk, nthreads, b, c_in, hin, win, c_out, hout, wout, group_in);
                                        // ,w1,w2,w3);
}