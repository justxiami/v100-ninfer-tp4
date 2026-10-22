#include "core/prefill_profile.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace ninfer::profile {
namespace {

constexpr int kSegCount = static_cast<int>(Seg::Count);
constexpr int kCapacity = 4096;

constexpr const char* kNames[kSegCount] = {
    "embed",         "norm_in",      "gdn_ctrl_proj", "gdn_in_proj",  "gdn_conv",
    "gdn_core",      "gdn_norm_out", "gdn_out_proj",  "attn_proj",    "attn_norm_rope",
    "attn_core",     "attn_gate",    "attn_out_proj", "allreduce",    "norm_post",
    "mlp_gate_up",   "mlp_down",     "final_norm",    "lm_head",      "sample",
    "mtp_stem",      "mtp_kv",       "mtp_tail",      "dflash_feat",  "dflash_layers",
};

struct Rec {
    Seg seg;
    cudaEvent_t start;
    cudaEvent_t end;
};

struct State {
    bool initialised = false;
    bool enabled     = false;
    cudaStream_t stream = nullptr;

    std::vector<cudaEvent_t> free_events;
    std::vector<Rec> live;

    double totals_us[kSegCount]  = {};
    long counts[kSegCount]       = {};
    double wall_us               = 0.0;
    cudaEvent_t wall_start       = nullptr;
    cudaEvent_t wall_end         = nullptr;
    bool wall_open               = false;

    std::int32_t tokens = 0;
    std::int32_t base   = 0;
    std::int32_t ranks  = 0;
    long chunks         = 0;
    FILE* out           = nullptr;
};

State& state() {
    static State s;
    return s;
}

void initialise() {
    State& s       = state();
    s.initialised  = true;
    const char* on = std::getenv("NINFER_PREFILL_PROFILE");
    s.enabled      = on != nullptr && std::strcmp(on, "0") != 0;
    if (!s.enabled) { return; }
    const char* path = std::getenv("NINFER_PREFILL_PROFILE_OUT");
    s.out            = path != nullptr ? std::fopen(path, "a") : stderr;
    if (s.out == nullptr) { s.out = stderr; }
    (void)cudaEventCreateWithFlags(&s.wall_start, cudaEventDefault);
    (void)cudaEventCreateWithFlags(&s.wall_end, cudaEventDefault);
}

[[nodiscard]] cudaEvent_t take_event() {
    State& s = state();
    if (!s.free_events.empty()) {
        cudaEvent_t ev = s.free_events.back();
        s.free_events.pop_back();
        return ev;
    }
    cudaEvent_t ev = nullptr;
    (void)cudaEventCreateWithFlags(&ev, cudaEventDefault);
    return ev;
}

// Adds every live record to the table and recycles its events. The caller has already made the
// stream's work complete (end_chunk synchronises), so elapsed_time cannot report not-ready.
void drain() {
    State& s = state();
    for (const Rec& rec : s.live) {
        float ms = 0.0F;
        if (cudaEventElapsedTime(&ms, rec.start, rec.end) == cudaSuccess) {
            const auto slot = static_cast<int>(rec.seg);
            s.totals_us[slot] += static_cast<double>(ms) * 1000.0;
            s.counts[slot] += 1;
        }
        s.free_events.push_back(rec.start);
        s.free_events.push_back(rec.end);
    }
    s.live.clear();
}

void write_row(const char* name, long count, double us, double wall_us) {
    State& s = state();
    const double pct = wall_us > 0.0 ? 100.0 * us / wall_us : 0.0;
    std::fprintf(s.out, "%-14s %6ld %12.1f %7.2f%%\n", name, count, us, pct);
}

} // namespace

const char* seg_name(Seg seg) noexcept {
    const auto slot = static_cast<int>(seg);
    return slot >= 0 && slot < kSegCount ? kNames[slot] : "?";
}

bool seg_is_nested(Seg seg) noexcept { return seg == Seg::AllReduce; }

bool enabled() noexcept {
    State& s = state();
    if (!s.initialised) { initialise(); }
    return s.enabled;
}

cudaStream_t profile_stream() noexcept { return state().stream; }

void set_profile_stream(cudaStream_t stream) noexcept {
    if (!enabled()) { return; }
    state().stream = stream;
}

Scope::Scope(Seg seg, cudaStream_t stream) noexcept : seg_(seg) {
    State& s = state();
    if (!s.enabled || s.stream == nullptr || stream != s.stream) { return; }
    if (static_cast<int>(s.live.size()) >= kCapacity) { return; }
    Rec rec{seg, take_event(), take_event()};
    if (rec.start == nullptr || rec.end == nullptr) { return; }
    if (cudaEventRecord(rec.start, stream) != cudaSuccess) { return; }
    s.live.push_back(rec);
    index_ = s.live.size() - 1;
    armed_ = true;
}

void Scope::stop() noexcept {
    if (!armed_) { return; }
    State& s = state();
    (void)cudaEventRecord(s.live[index_].end, s.stream);
    armed_ = false;
}

Scope::~Scope() { stop(); }

void begin_chunk(std::int32_t tokens, std::int32_t base, std::int32_t rank_count) {
    State& s = state();
    if (!enabled()) { return; }
    if (s.wall_open) { end_chunk(); }
    drain();
    for (double& total : s.totals_us) { total = 0.0; }
    for (long& count : s.counts) { count = 0; }
    s.wall_us    = 0.0;
    s.tokens     = tokens;
    s.base       = base;
    s.ranks      = rank_count;
    s.wall_open  = true;
    (void)cudaEventRecord(s.wall_start, s.stream);
}

void end_chunk() {
    State& s = state();
    if (!s.enabled || !s.wall_open) { return; }
    (void)cudaEventRecord(s.wall_end, s.stream);
    (void)cudaStreamSynchronize(s.stream);
    float wall_ms = 0.0F;
    if (cudaEventElapsedTime(&wall_ms, s.wall_start, s.wall_end) == cudaSuccess) {
        s.wall_us = static_cast<double>(wall_ms) * 1000.0;
    }
    drain();
    s.wall_open = false;
    ++s.chunks;

    std::fprintf(s.out, "# prefill-chunk seq=%ld tokens=%d base=%d ranks=%d\n", s.chunks + 1,
                 s.tokens, s.base, s.ranks);
    std::fprintf(s.out, "# %-12s %6s %12s %8s\n", "segment", "count", "total_us", "wall%");
    double nested_us = 0.0;
    double top_us    = 0.0;
    for (int i = 0; i < kSegCount; ++i) {
        if (s.counts[i] == 0) { continue; }
        write_row(kNames[i], s.counts[i], s.totals_us[i], s.wall_us);
        if (seg_is_nested(static_cast<Seg>(i))) {
            nested_us += s.totals_us[i];
        } else {
            top_us += s.totals_us[i];
        }
    }
    std::fprintf(s.out, "%-14s %6d %12.1f %7.2f%%\n", "wall", 1, s.wall_us, 100.0);
    std::fprintf(s.out, "%-14s %6s %12.1f %7.2f%%\n", "top_sum", "-", top_us,
                 s.wall_us > 0.0 ? 100.0 * top_us / s.wall_us : 0.0);
    std::fprintf(s.out, "%-14s %6s %12.1f %7.2f%%\n", "unaccounted", "-", s.wall_us - top_us,
                 s.wall_us > 0.0 ? 100.0 * (s.wall_us - top_us) / s.wall_us : 0.0);
    std::fprintf(s.out, "%-14s %6s %12.1f %7.2f%%\n", "(nested AR)", "-", nested_us,
                 s.wall_us > 0.0 ? 100.0 * nested_us / s.wall_us : 0.0);
    std::fflush(s.out);
}

} // namespace ninfer::profile
