#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstring>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto* model = llama_model_load_from_file(argv[1], mp); if (!model) return 3;
    auto* vocab = llama_model_get_vocab(model);
    auto cp = llama_context_default_params(); cp.n_ctx=512; cp.n_batch=512; cp.n_ubatch=512;
    cp.n_threads=12; cp.n_threads_batch=12;
    auto* ctx=llama_init_from_model(model,cp); if (!ctx) return 4;
    const char* prompt="The quick brown fox";
    std::vector<llama_token> tokens(-llama_tokenize(vocab,prompt,strlen(prompt),nullptr,0,true,true));
    llama_tokenize(vocab,prompt,strlen(prompt),tokens.data(),tokens.size(),true,true);
    printf("{\"promptTokens\":[");
    for (size_t i=0;i<tokens.size();i++) printf("%s%d",i?",":"",tokens[i]);
    printf("],\"steps\":[");
    auto batch=llama_batch_get_one(tokens.data(),tokens.size());
    int size=llama_vocab_n_tokens(vocab);
    for(int step=0;step<8;step++) {
        if(llama_decode(ctx,batch))return 5;
        float* logits=llama_get_logits_ith(ctx,-1);
        std::vector<int> ids(size);for(int i=0;i<size;i++)ids[i]=i;
        std::partial_sort(ids.begin(),ids.begin()+10,ids.end(),[&](int a,int b){return logits[a]>logits[b];});
        printf("%s{\"token\":%d,\"topLogits\":[",step?",":"",ids[0]);
        for(int i=0;i<10;i++)printf("%s[%d,%.9g]",i?",":"",ids[i],logits[ids[i]]);
        printf("]}");fflush(stdout);
        tokens={ids[0]};batch=llama_batch_get_one(tokens.data(),1);
    }
    puts("]}");llama_free(ctx);llama_model_free(model);llama_backend_free();
}
