// External qualification harness only. Links an independently built llama.cpp, never Models.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
using clock_type = std::chrono::steady_clock;
static double ms(clock_type::time_point start) { return std::chrono::duration<double, std::milli>(clock_type::now()-start).count(); }
template<class T> static std::vector<T> read(const char *path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("missing input");
    auto size = f.tellg();
    if (size <= 0 || size % sizeof(T)) throw std::runtime_error("invalid input size");
    std::vector<T> v(size / sizeof(T)); f.seekg(0); f.read(reinterpret_cast<char *>(v.data()), size);
    if (!f) throw std::runtime_error("short input"); return v;
}
int main(int argc, char **argv) {
    if (argc != 9) { std::cerr << "model embeddings prefix suffix threads warmups repetitions batch\n"; return 2; }
    auto embeddings=read<float>(argv[2]); auto prefix=read<llama_token>(argv[3]); auto suffix=read<llama_token>(argv[4]);
    int threads=std::stoi(argv[5]), warmups=std::stoi(argv[6]), repetitions=std::stoi(argv[7]), capacity=std::stoi(argv[8]);
    if (threads<1 || warmups<1 || repetitions<3 || capacity<1) return 2;
    llama_backend_init(); auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    auto model=llama_model_load_from_file(argv[1],mp); if (!model) return 3;
    auto cp=llama_context_default_params(); cp.n_ctx=1024; cp.n_batch=capacity; cp.n_ubatch=capacity;
    cp.n_threads=threads; cp.n_threads_batch=threads;
    auto ctx=llama_init_from_model(model,cp); if (!ctx) return 3;
    int dim=llama_model_n_embd(model); auto vocab=llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(vocab);
    if (embeddings.size()%dim) return 2;
    auto tb=llama_batch_init(capacity,0,1); auto eb=llama_batch_init(capacity,dim,1);
    for (int run=-warmups; run<repetitions; ++run) {
        llama_memory_clear(llama_get_memory(ctx),true); int pos=0;
        auto start=clock_type::now();
        auto submit=[&](llama_batch &b,int count,bool final) {
            b.n_tokens=count;
            for (int i=0;i<count;++i) { b.pos[i]=pos++; b.n_seq_id[i]=1; b.seq_id[i][0]=0; b.logits[i]=final && i==count-1; }
            if (llama_decode(ctx,b)) throw std::runtime_error("decode failed");
        };
        auto tokens=[&](const std::vector<llama_token> &v,bool final) {
            for (size_t offset=0;offset<v.size();offset+=capacity) {
                int n=std::min<size_t>(capacity,v.size()-offset); std::copy_n(v.data()+offset,n,tb.token);
                submit(tb,n,final && offset+n==v.size());
            }
        };
        tokens(prefix,false);
        for (size_t offset=0;offset<embeddings.size()/dim;offset+=capacity) {
            int n=std::min<size_t>(capacity,embeddings.size()/dim-offset);
            std::copy_n(embeddings.data()+offset*dim,n*dim,eb.embd); submit(eb,n,false);
        }
        tokens(suffix,true); auto logits=llama_get_logits_ith(ctx,-1);
        double prefill=ms(start); start=clock_type::now(); std::vector<int> generated;
        if (run==0) { std::ofstream f("oracle-first-logits.f32",std::ios::binary); f.write(reinterpret_cast<char *>(logits),nv*sizeof(float)); }
        for (int i=0;i<32;i++) {
            for (int j=0;j<nv;j++) if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite logits");
            int best=std::max_element(logits,logits+nv)-logits; if (llama_vocab_is_eog(vocab,best)) break;
            generated.push_back(best); if (i<31) { tb.token[0]=best; submit(tb,1,true); logits=llama_get_logits_ith(ctx,-1); }
        }
        double decode=ms(start);
        if (run>=0) {
            std::cout<<"{\"run\":"<<run<<",\"prefillMs\":"<<prefill<<",\"decodeMs\":"<<decode<<",\"tokens\":[";
            for (size_t i=0;i<generated.size();i++) std::cout<<(i?",":"")<<generated[i];
            std::cout<<"]}"<<std::endl;
        }
    }
    llama_batch_free(tb); llama_batch_free(eb); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
