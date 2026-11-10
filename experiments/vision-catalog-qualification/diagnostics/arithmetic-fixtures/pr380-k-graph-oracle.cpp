// Independent graph oracle: ordinary CPU buffers dispatch Q5/Q6 to tiled kernels at batch >= 8.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <vector>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <iostream>
#include <cstdint>
uint32_t state=371;
std::vector<float> values(size_t n){std::vector<float>x(n);for(auto &v:x){state=1664525*state+1013904223;v=float(state>>8)*(1.f/16777216.f)*8.f-4.f;}return x;}
void save(const std::filesystem::path&p,const void*data,size_t n){std::ofstream f(p,std::ios::binary);f.write((const char*)data,n);if(!f)throw std::runtime_error("write");}
int main(int argc,char**argv){if(argc!=2)return 2;ggml_cpu_init();auto root=std::filesystem::path(argv[1]);std::filesystem::create_directories(root);
for(auto type:{GGML_TYPE_Q5_K,GGML_TYPE_Q6_K})for(int cols:{768,4096}){
 auto dir=root/(type==GGML_TYPE_Q5_K?"q5":"q6")/std::to_string(cols);std::filesystem::create_directories(dir);
 auto weights=values(cols*8),input=values(cols*17);std::vector<char>packed(ggml_row_size(type,cols)*8);
 auto n=ggml_quantize_chunk(type,weights.data(),packed.data(),0,8,cols,nullptr);if(n!=packed.size())throw std::runtime_error("size");save(dir/"weights.bin",packed.data(),packed.size());save(dir/"input.f32",input.data(),input.size()*4);
 for(int batch:{1,3,7,8,17}){
  auto ctx=ggml_init({16*1024*1024,nullptr,false});auto w=ggml_new_tensor_2d(ctx,type,cols,8);memcpy(w->data,packed.data(),packed.size());auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,batch);memcpy(x->data,input.data(),ggml_nbytes(x));
  auto y=ggml_mul_mat(ctx,w,x);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);if(ggml_graph_compute_with_ctx(ctx,graph,1)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");save(dir/("expected-"+std::to_string(batch)+".f32"),y->data,ggml_nbytes(y));ggml_free(ctx);
 }std::cout<<dir<<std::endl;
}}
