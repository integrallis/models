// Independent fixture generator: ggml 2cbca389f515df455412dafa07e2b1359cc72a6d.
// External test oracle only, never linked into Models.
#include "ggml.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <initializer_list>
#include <cstring>
#include <cmath>
int main(int argc,char**argv) {
 if(argc!=2)return 2;
 ggml_cpu_init();
 for(int mode: {0,2}) {
  ggml_init_params params{4*1024*1024,nullptr,false}; auto ctx=ggml_init(params);
  auto x=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,128,2,5);
  auto pos=ggml_new_tensor_1d(ctx,GGML_TYPE_I32,5);int positions[]={0,1,31,639,4095};std::memcpy(pos->data,positions,sizeof(positions));
  for(int i=0;i<128*2*5;i++)static_cast<float*>(x->data)[i]=float(i%19-9)*.125f;
  auto result=ggml_rope_ext(ctx,x,pos,nullptr,128,mode,1024,10000.f,.25f,0.f,1.f,32.f,1.f);
  auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,result);
  if(ggml_graph_compute_with_ctx(ctx,graph,1)!=GGML_STATUS_SUCCESS)return 3;
  char name[1024];std::snprintf(name,sizeof(name),"%s/rope-%d.f32",argv[1],mode);auto f=std::fopen(name,"wb");if(!f)return 4;std::fwrite(result->data,4,128*2*5,f);std::fclose(f);ggml_free(ctx);
 }
 // Record the oracle's float factors separately so rotation can be tested bitwise
 // independently of Java double versus native float transcendental functions.
 volatile float base = 10000.f;
 const float step = std::pow(base, -2.f / 128.f);
 float factors[5 * 128];
 int positions[] = {0, 1, 31, 639, 4095};
 for (int row = 0; row < 5; ++row) {
  float angle = static_cast<float>(positions[row]);
  for (int pair = 0; pair < 64; ++pair) {
   factors[row * 128 + pair * 2] = std::cos(angle * .25f);
   factors[row * 128 + pair * 2 + 1] = std::sin(angle * .25f);
   angle *= step;
  }
 }
 char name[1024];
 std::snprintf(name, sizeof(name), "%s/factors.f32", argv[1]);
 auto file = std::fopen(name, "wb");
 if (!file) return 4;
 std::fwrite(factors, sizeof(float), 5 * 128, file);
 std::fclose(file);

}
