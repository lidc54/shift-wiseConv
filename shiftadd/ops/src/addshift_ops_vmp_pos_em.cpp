#include <torch/extension.h>

void launch_shift_add_forward_v_mp_linear(float *out, const float *input,
                                const int *pad_hv, const int *idx_identit, const int *idx_out, 
                                int extra_pad, int b, int c_in, int hin, int win,
                                int c_out, int hout, int wout, int group_in,
                                const float *w1, float *w2, float *w3);

void launch_shift_add_backward_v_mp_linear(float *gout, float *ginput,
                                const int *pad_hv, const int *idx_identit, const int *idx_out, 
                                int extra_pad, int b, int c_in, int hin, int win,
                                int c_out, int hout, int wout, int group_in);
                              //   ,float *w1, float *w2, float *w3);

void add_forward_v_mp_linear(torch::Tensor &out,
                          const torch::Tensor &input,
                          const torch::Tensor &pad_hv,
                          const torch::Tensor &idx_identit,
                          const torch::Tensor &idx_out,
                          int64_t extra_pad, int64_t b, int64_t c_in, int64_t hin, int64_t win, 
                          int64_t c_out, int64_t hout, int64_t wout, int64_t group_in,
                          torch::Tensor &w1, torch::Tensor &w2, torch::Tensor &w3)
{
    /*out 需要使用torch chunk 分开为group组*/
    launch_shift_add_forward_v_mp_linear((float *)out.data_ptr(),
                    (const float *)input.data_ptr(),
                    (const int *)pad_hv.data_ptr(),
                    (const int *)idx_identit.data_ptr(),
                    (const int *)idx_out.data_ptr(),
                    extra_pad, b, c_in, hin, win,
                    c_out, hout, wout, group_in,
                    // (float *)w1.data_ptr(), 
                    (const float *)w1.data_ptr(), 
                    // w1.data_ptr<float>(), 
                    (float *)w2.data_ptr(), 
                    (float *)w3.data_ptr());
}

void add_backward_v_mp_linear(torch::Tensor &out,
                          torch::Tensor &input,
                          const torch::Tensor &pad_hv,
                          const torch::Tensor &idx_identit,
                          const torch::Tensor &idx_out,
                          int64_t extra_pad, int64_t b, int64_t c_in, int64_t hin, int64_t win, 
                          int64_t c_out, int64_t hout, int64_t wout, int64_t group_in)
                        //   ,torch::Tensor &w1, torch::Tensor &w2, torch::Tensor &w3)
{
    launch_shift_add_backward_v_mp_linear((float *)out.data_ptr(),
                    (float *)input.data_ptr(),
                    (const int *)pad_hv.data_ptr(),
                    (const int *)idx_identit.data_ptr(),
                    (const int *)idx_out.data_ptr(),
                    extra_pad, b, c_in, hin, win,
                    c_out, hout, wout, group_in); 
                  //   ,(float *)gw1.data_ptr(), 
                  //   (float *)gw2.data_ptr(), 
                  //   (float *)gw3.data_ptr());                      
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward",
          &add_forward_v_mp_linear,
          "shift_add_v_mp_linear kernel warpper");

    m.def("backward",
          &add_backward_v_mp_linear,
          "shift_add_v_mp_linear kernel warpper");
}
