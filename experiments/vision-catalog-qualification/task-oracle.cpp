// External qualification harness only. Links an independently built llama.cpp, never Models.
#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sstream>
#include <iomanip>
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
    if (argc != 4 && argc != 5) { std::cerr << "model cases.tsv threads [f32|f16]\n"; return 2; }
    int threads=std::stoi(argv[3]), capacity=128;
    if (threads<1) return 2;
    std::string kv=argc==5 ? argv[4] : "f32";
    if (kv!="f32" && kv!="f16") return 2;
    auto loaded=clock_type::now();
    llama_backend_init(); auto mp=llama_model_default_params(); mp.n_gpu_layers=0;
    auto model=llama_model_load_from_file(argv[1],mp); if (!model) return 3;
    auto cp=llama_context_default_params(); cp.n_ctx=1024; cp.n_batch=capacity; cp.n_ubatch=capacity;
    cp.n_threads=threads; cp.n_threads_batch=threads;
    cp.type_k=kv=="f32" ? GGML_TYPE_F32 : GGML_TYPE_F16; cp.type_v=cp.type_k;
    // Preserve the frozen reference route explicitly. AUTO casts F32 cache values
    // to F16 inside Flash Attention; kvType describes storage, not arithmetic.
    cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_AUTO;
    auto ctx=llama_init_from_model(model,cp); if (!ctx) return 3;
    int dim=llama_model_n_embd(model); auto vocab=llama_model_get_vocab(model); int nv=llama_vocab_n_tokens(vocab);
    auto tb=llama_batch_init(capacity,0,1); auto eb=llama_batch_init(capacity,dim,1);
    std::cout << "{\"configuration\":{\"threads\":" << threads
      << ",\"context\":1024,\"maximumOutputTokens\":32,\"batch\":" << capacity
      << ",\"kvType\":\"" << kv << "\",\"flashAttention\":\"" << "auto" << "\",\"loadMs\":" << ms(loaded) << "}}" << std::endl;
    std::ifstream cases(argv[2]); if (!cases) throw std::runtime_error("missing cases");
    std::string line;
    while (std::getline(cases,line)) {
        std::istringstream fields(line);
        std::string id, embeddingPath, prefixPath, suffixPath; int warmups, repetitions;
        if (!(fields >> id >> embeddingPath >> prefixPath >> suffixPath >> warmups >> repetitions))
            throw std::runtime_error("invalid case line");
        if (warmups<0 || repetitions<1) throw std::runtime_error("invalid repetitions");
        if (id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-") != std::string::npos)
            throw std::runtime_error("invalid case ID");
        auto embeddings=read<float>(embeddingPath.c_str());
        auto prefix=read<llama_token>(prefixPath.c_str()); auto suffix=read<llama_token>(suffixPath.c_str());
        if (embeddings.size()%dim || prefix.size()+suffix.size()+embeddings.size()/dim+32>1024)
            throw std::runtime_error("invalid dimensions/context");
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
        for (int i=0;i<32;i++) {
            for (int j=0;j<nv;j++) if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite logits");
            int best=std::max_element(logits,logits+nv)-logits; if (llama_vocab_is_eog(vocab,best)) break;
            generated.push_back(best); if (i<31) { tb.token[0]=best; submit(tb,1,true); logits=llama_get_logits_ith(ctx,-1); }
        }
        double decode=ms(start);
        {
            std::cout<<"{\"id\":\""<<id<<"\",\"run\":"<<run<<",\"prefillMs\":"<<prefill<<",\"decodeMs\":"<<decode<<",\"tokens\":[";
            for (size_t i=0;i<generated.size();i++) std::cout<<(i?",":"")<<generated[i];
            std::string answer;
            for (auto token : generated) {
                std::vector<char> piece(256);
                int count=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,false);
                if (count<0) { piece.resize(-count); count=llama_token_to_piece(vocab,token,piece.data(),piece.size(),0,false); }
                if (count<0) throw std::runtime_error("token decode failed");
                answer.append(piece.data(),count);
            }
            std::cout<<"],\"answerHex\":\"";
            for (unsigned char c : answer) std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c);
            std::cout<<std::dec<<"\"}"<<std::endl;
        }
    }
    }
    llama_batch_free(tb); llama_batch_free(eb); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
