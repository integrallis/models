// External llama.cpp diagnostic only. Never linked into a Models runtime or distribution.
#include "llama.h"
#include "ggml-backend.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

template<class T> static std::vector<T> read(const char *path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("missing input");
    auto size = f.tellg();
    if (size <= 0 || size % sizeof(T)) throw std::runtime_error("invalid input size");
    std::vector<T> result(size / sizeof(T)); f.seekg(0);
    f.read(reinterpret_cast<char *>(result.data()), size);
    if (!f) throw std::runtime_error("short input");
    return result;
}

static void write(const std::filesystem::path &path, const float *values, size_t count) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(values), count * sizeof(float));
    if (!f) throw std::runtime_error("trace write failed");
}

struct Trace { std::filesystem::path output; bool active = false; int position = 0; int dimension = 0; int batch = 0; };
static bool observe(ggml_tensor *tensor, bool ask, void *data) {
    auto &trace = *static_cast<Trace *>(data);
    std::string name = tensor->name;
    bool cache = name=="Qcur-0" || name=="Kcur-0" || name=="Vcur-0";
    bool selected = cache || trace.active && (name.rfind("l_out-", 0) == 0 ||
        (name.size()>2 && name.substr(name.size()-2)=="-0"));
    if (ask) return selected;
    if (!selected) return true;
    if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float)) return true;
    std::vector<float> values(ggml_nbytes(tensor)/sizeof(float));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
    if (cache) {
        if (!ggml_is_contiguous(tensor)) throw std::runtime_error("noncontiguous QKV");
        auto path = trace.output / (name+"-all.f32");
        if (!std::filesystem::exists(path)) { std::ofstream create(path, std::ios::binary); }
        std::fstream all(path, std::ios::in | std::ios::out | std::ios::binary);
        std::ofstream metadata(trace.output / "qkv-chunks.jsonl", std::ios::app);
        metadata << "{\"tensor\":\"" << name << "\",\"position\":" << trace.position
            << ",\"batch\":" << trace.batch << ",\"shape\":[" << tensor->ne[0] << "," << tensor->ne[1] << "," << tensor->ne[2]
            << "],\"values\":" << values.size() << "}\n";
        if (values.size() != static_cast<size_t>(trace.dimension)*trace.batch) throw std::runtime_error("QKV chunk shape");
        all.seekp(static_cast<long long>(trace.position)*trace.dimension*sizeof(float));
        all.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
        if (!all) throw std::runtime_error("QKV trace write failed");
    }
    if (!trace.active) return true;
    write(trace.output / (name + ".f32"), values.data(), values.size());
    std::ofstream shape(trace.output / (name + ".shape.json"));
    shape << "[" << tensor->ne[0] << "," << tensor->ne[1] << "," << tensor->ne[2] << "," << tensor->ne[3] << "]";
    if (name.rfind("l_out-", 0)==0) {
        std::vector<float> last(tensor->ne[0]);
        ggml_backend_tensor_get(tensor, last.data(), (tensor->ne[1]-1)*tensor->nb[1], last.size()*sizeof(float));
        write(trace.output / ("layer-"+name.substr(6)+".f32"), last.data(), last.size());
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc != 6 && argc != 7) { std::cerr << "MODEL PREFIX.i32 SUFFIX.i32 EMBEDDINGS.f32 NEW_OUTPUT_DIR [auto|off]\n"; return 2; }
    Trace trace{argv[5]};
    if (!std::filesystem::create_directory(trace.output)) throw std::runtime_error("preserve prior output");
    auto prefix = read<llama_token>(argv[2]), suffix = read<llama_token>(argv[3]);
    auto embeddings = read<float>(argv[4]);
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    auto model = llama_model_load_from_file(argv[1], mp);
    if (!model) throw std::runtime_error("load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 1024; cp.n_batch = 128; cp.n_ubatch = 128;
    cp.n_threads = 8; cp.n_threads_batch = 8;
    cp.type_k = GGML_TYPE_F32; cp.type_v = GGML_TYPE_F32;
    if (argc == 7) {
        std::string attention = argv[6];
        if (attention != "auto" && attention != "off") throw std::runtime_error("attention mode");
        if (attention == "off") cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    cp.cb_eval = observe; cp.cb_eval_user_data = &trace;
    trace.dimension = llama_model_n_embd(model);
    auto ctx = llama_init_from_model(model, cp);
    if (!ctx) throw std::runtime_error("context failed");
    int dim = llama_model_n_embd(model), pos = 0;
    if (embeddings.size() % dim || prefix.size() + suffix.size() + embeddings.size()/dim > 1024)
        throw std::runtime_error("invalid context shape");
    auto tb = llama_batch_init(128, 0, 1), eb = llama_batch_init(128, dim, 1);
    auto submit = [&](llama_batch &batch, int n, bool final) {
        batch.n_tokens = n; trace.active = final; trace.position = pos; trace.batch = n;
        for (int i=0; i<n; ++i) {
            batch.pos[i] = pos++; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = final && i==n-1;
        }
        if (llama_decode(ctx, batch)) throw std::runtime_error("decode failed");
    };
    auto tokens = [&](const std::vector<llama_token> &values, bool final) {
        for (size_t offset=0; offset<values.size(); offset+=128) {
            int n = std::min<size_t>(128, values.size()-offset);
            std::copy_n(values.data()+offset, n, tb.token);
            submit(tb, n, final && offset+n==values.size());
        }
    };
    tokens(prefix, false);
    for (size_t offset=0; offset<embeddings.size()/dim; offset+=128) {
        int n = std::min<size_t>(128, embeddings.size()/dim-offset);
        std::copy_n(embeddings.data()+offset*dim, n*dim, eb.embd); submit(eb, n, false);
    }
    tokens(suffix, true);
    auto logits = llama_get_logits_ith(ctx, -1);
    int count = llama_vocab_n_tokens(llama_model_get_vocab(model));
    write(trace.output / "logits.f32", logits, count);
    int best = std::max_element(logits, logits+count)-logits;
    std::ofstream summary(trace.output / "prediction.json");
    summary << "{\"position\":" << pos-1 << ",\"topToken\":" << best << ",\"logit\":" << logits[best] << "}\n";
    llama_batch_free(tb); llama_batch_free(eb); llama_free(ctx); llama_model_free(model); llama_backend_free();
}
