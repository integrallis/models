// Synthetic independent oracle only. Pinned ggml is never linked into Models.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <fstream>
#include <vector>
#include <string>
#include <filesystem>
#include <stdexcept>
#include <limits>
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 std::filesystem::create_directories(argv[1]);
 uint32_t seed=371;
 auto next=[&](){seed=1664525u*seed+1013904223u;return float(seed>>8)*(1.0f/16777216.0f)*0.5f-0.25f;};
 auto save=[&](std::string name,const float*p,size_t n){std::ofstream f(std::string(argv[1])+"/"+name,std::ios::binary);f.write((const char*)p,n*4);if(!f)throw std::runtime_error("write failed");};
 ggml_cpu_init();
 for(int n:{7,32,4096,8193}) {
  ggml_init_params params{4*1024*1024,nullptr,false};auto ctx=ggml_init(params);
  auto x=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);auto w=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
  auto xp=(float*)x->data;auto wp=(float*)w->data;
  for(int i=0;i<n;i++)xp[i]=next()*64;
  const float edges[]={0.f,-0.f,100.f,-100.f,89.f,-89.f,87.f,-87.f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()};
  for(int i=0;i<n && i<11;i++)xp[i]=edges[i];for(int i=0;i<n;i++)wp[i]=next();
  auto result=ggml_mul(ctx,ggml_silu(ctx,x),w);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,result);
  if(ggml_graph_compute_with_ctx(ctx,graph,3)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute failed");
  save(std::to_string(n)+"-input.f32",xp,n);save(std::to_string(n)+"-weight.f32",wp,n);save(std::to_string(n)+"-expected.f32",(float*)result->data,n);ggml_free(ctx);
 }
}
