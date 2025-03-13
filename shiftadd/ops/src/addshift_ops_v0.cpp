#include <torch/extension.h>
// #include "add2.h"

void launch_shift_add_forward_v0(float *out, const float *input,
                       const int *idxes, const int *shift_pads, int extra_pad,
                       int b, int c_in, int hin, int win,
                       int c_out, int hout, int wout, bool isH);

void launch_shift_add_backward_v0(float *out, float *input,
                       const int *idxes, const int *shift_pads, int extra_pad,
                       int b, int c_in, int hin, int win,
                       int c_out, int hout, int wout, bool isH);


void add_forward_v0(torch::Tensor &out,
                          const torch::Tensor &input,
                          const torch::Tensor &idxes,
                          const torch::Tensor &shift_pads,
                          int64_t extra_pad, int64_t b, int64_t c_in, int64_t hin, int64_t win, 
                          int64_t c_out, int64_t hout, int64_t wout, bool isH)
{
    launch_shift_add_forward_v0((float *)out.data_ptr(),
                    (const float *)input.data_ptr(),
                    (const int *)idxes.data_ptr(),
                    (const int *)shift_pads.data_ptr(),
                    extra_pad, b, c_in, hin, win,
                    c_out, hout, wout, isH);                      
}

void add_backward_v0(torch::Tensor &out,
                          torch::Tensor &input,
                          const torch::Tensor &idxes,
                          const torch::Tensor &shift_pads,
                          int64_t extra_pad, int64_t b, int64_t c_in, int64_t hin, int64_t win, 
                          int64_t c_out, int64_t hout, int64_t wout, bool isH)
{
    launch_shift_add_backward_v0((float *)out.data_ptr(),
                    (float *)input.data_ptr(),
                    (const int *)idxes.data_ptr(),
                    (const int *)shift_pads.data_ptr(),
                    extra_pad, b, c_in, hin, win,
                    c_out, hout, wout, isH);                      
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward",
          &add_forward_v0,
          "shift_add_v0 kernel warpper");

    m.def("backward",
          &add_backward_v0,
          "shift_add_v0 kernel warpper");
}

// TORCH_LIBRARY(add2_v0, m)
// {
//     m.def("add_forward_v0", add_forward_v0);
//     m.def("add_backward_v0", add_backward_v0);
// }


