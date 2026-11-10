// External synthetic oracle; uses the pinned runtime's actual CPU_REPACK buffer.
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "repack.h"
#include <vector>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <iostream>
std::vector<char> read(std::filesystem::path p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("read");std::vector<char>b(f.tellg());f.seekg(0);f.read(b.data(),b.size());return b;}
int main(int argc,char **argv){
 if(argc!=3)return 2;ggml_cpu_init();auto root=std::filesystem::path(argv[1]);auto out=std::filesystem::path(argv[2]);std::filesystem::create_directory(out);
 for(int cols:{256,768,4096,11008}){
  auto weights=read(root/std::to_string(cols)/"weights.q4k");auto inputs=read(root/std::to_string(cols)/"input.f32");
  ggml_init_params wp{1024*1024,nullptr,true};auto wc=ggml_init(wp);auto w=ggml_new_tensor_2d(wc,GGML_TYPE_Q4_K,cols,8);
  auto buffer=ggml_backend_alloc_ctx_tensors_from_buft(wc,ggml_backend_cpu_repack_buffer_type());if(!buffer)throw std::runtime_error("buffer");
  ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
  std::cout<<cols<<" "<<ggml_backend_buffer_name(buffer)<<" extra="<<(w->extra!=nullptr)<<std::endl;if(!w->extra)throw std::runtime_error("no repack traits");
  for(int batch:{1,3,4,8,17}){
   ggml_init_params params{16*1024*1024,nullptr,false};auto ctx=ggml_init(params);auto x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,cols,batch);memcpy(x->data,inputs.data(),ggml_nbytes(x));
   auto y=ggml_mul_mat(ctx,w,x);auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,y);
   if(ggml_graph_compute_with_ctx(ctx,graph,1)!=GGML_STATUS_SUCCESS)throw std::runtime_error("compute");
   auto p=out/(std::to_string(cols)+"-"+std::to_string(batch)+".f32");std::ofstream f(p,std::ios::binary);f.write((char*)y->data,ggml_nbytes(y));if(!f)throw std::runtime_error("write");ggml_free(ctx);
  }
  ggml_backend_buffer_free(buffer);ggml_free(wc);
 }
}
