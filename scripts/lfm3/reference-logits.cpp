// Copyright 2026 Integrallis Software, LLC. SPDX-License-Identifier: Apache-2.0
#include "llama.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
static bool capture(ggml_tensor *t,bool ask,void *) {
 const char *dir=std::getenv("LFM3_TRACE");
 bool keep=dir && t->type==GGML_TYPE_F32 && ggml_is_contiguous(t) && ggml_nbytes(t)<4000000;
 if(ask)return keep;
 if(keep){std::vector<char> data(ggml_nbytes(t));ggml_backend_tensor_get(t,data.data(),0,data.size());std::ofstream out(std::string(dir)+"/"+t->name+".bin",std::ios::binary);out.write(data.data(),data.size());}
 return true;
}
int main(int argc,char **argv) {
 if(argc!=4){std::cerr<<"MODEL TOKEN_CSV OUTPUT\n";return 2;}
 llama_backend_init();auto mp=llama_model_default_params();mp.n_gpu_layers=0;
 auto *model=llama_model_load_from_file(argv[1],mp);if(!model)return 3;
 auto cp=llama_context_default_params();cp.n_ctx=2048;cp.n_batch=512;cp.n_ubatch=512;
 cp.cb_eval=capture;
 cp.n_threads=4;cp.n_threads_batch=4;cp.type_k=GGML_TYPE_F32;cp.type_v=GGML_TYPE_F32;
 cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_DISABLED;
 auto *ctx=llama_init_from_model(model,cp);if(!ctx)return 4;
 std::vector<llama_token> tokens;std::stringstream ss(argv[2]);std::string item;
 while(std::getline(ss,item,','))tokens.push_back(std::stoi(item));
 auto batch=llama_batch_get_one(tokens.data(),tokens.size());
 if(llama_decode(ctx,batch)!=0)return 5;
 auto *logits=llama_get_logits_ith(ctx,-1);int size=llama_vocab_n_tokens(llama_model_get_vocab(model));
 std::ofstream out(argv[3],std::ios::binary);out.write(reinterpret_cast<const char *>(logits),size*sizeof(float));
 llama_free(ctx);llama_model_free(model);llama_backend_free();return out?0:6;
}
