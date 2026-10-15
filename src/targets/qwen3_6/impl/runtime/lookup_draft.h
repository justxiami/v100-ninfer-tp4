#pragma once
// Host-side n-gram lookup drafter -- phase 1, DEBUG-GATED (2026-10-14).
//
// WHY HERE.  The MTP round already crosses the host boundary twice: the proposal for the NEXT
// round is read back from the device egress record, and written into the next round's PINNED
// ingress record.  `decode_graph.cpp` bakes the ingress record in by ADDRESS and reads its
// CONTENT at replay, so replacing that content costs no graph, no kernel and no numerics change.
// This file computes that content on the host from the token ledger the runtime already keeps for
// prefix reuse (`SequenceState::ledger`, filled by the prompt at program_impl.h:866 and appended
// per committed token at program_impl.h:1132).
//
// CORRECTNESS.  A proposal only decides what the target VERIFIES.  Tokens are committed by the
// existing accept path, so a wrong proposal costs verify width, never correctness.  This is why
// the feature can be evaluated as a pure host-side arm.
//
// ALGORITHM.  Keep every position whose m-token window is complete and is NOT the current suffix,
// in a fixed-size 4-way hash table (recent-wins on collision).  To propose: hash the last m
// tokens, walk that bucket, verify each candidate literally (the table only stores positions, so
// a bucket hit is not yet a match), prefer the candidate with the longest backward-context run,
// then copy up to k of the tokens that followed it before.  Proposals never invent a token: every
// id comes from the ledger, so it is in-vocabulary and in-domain by construction.
//
// Knobs (all off by default; one env read, cached):
//   NINFER_TP4_LOOKUP=1         master switch.  `0`/unset = zero cost, zero behaviour change.
//   NINFER_TP4_LOOKUP_MIN=3     suffix length m, clamped to [2, 8].
//   NINFER_TP4_LOOKUP_MIN_RUN=2 how many tokens BEFORE the matched copy must also agree. This is
//                               the spurious-hit filter: on code text a 3-token suffix matches by
//                               accident all the time (whitespace and brace runs), and the offline
//                               sweep (~/bench-results/2026-10-14/lookup_spec/SWEEP_SYNTH.txt)
//                               shows those accidental fires drag a non-repetitive stream from
//                               ~2.4 tok/round down to 1.35 if they are used unchecked.
//   NINFER_TP4_LOOKUP_KEEP=0    reserved (0 = fall back to MTP when the lookup misses).
//   NINFER_TP4_LOOKUP_MAX=0     cap on proposed drafts (0 = use the round's draft window).
//   NINFER_TP4_LOOKUP_DUMP=pfx  write <pfx>.req (per finished request; works with LOOKUP=0 too,
//                               which is how a baseline boot supplies a real token stream for
//                               the offline replay) and <pfx>.round (per round; LOOKUP=1 only).
//
// HYBRID BY CONSTRUCTION.  When the lookup finds no usable candidate the caller's existing
// proposal stays untouched, so the arm is "MTP, with the proposal replaced where history has a
// copy" rather than "lookup instead of MTP".

#include <ninfer/types.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::lookup {

using ninfer::TokenId;

inline constexpr std::uint32_t kMaxSuffix      = 8;
inline constexpr std::uint32_t kMaxProposal    = 16;
inline constexpr std::uint32_t kBackwardCap    = 16;
inline constexpr std::uint32_t kWaysPerBucket  = 4;
inline constexpr std::uint32_t kMinimumSlots   = 1U << 12;
inline constexpr std::uint32_t kMaximumSlots   = 1U << 19;

struct LookupRuntimeConfig {
    bool enabled        = false;
    std::uint32_t min_match = 3;
    std::uint32_t min_run   = 2;
    std::uint32_t max_drafts = 0;                 // 0 = the round's own draft window
    std::string dump_prefix;
};

[[nodiscard]] inline bool env_flag(const char* name, bool fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return fallback; }
    return !(*raw == '0' && raw[1] == '\0');
}

[[nodiscard]] inline std::uint32_t env_uint(const char* name, std::uint32_t fallback) {
    const char* raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') { return fallback; }
    char* end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw) { return fallback; }
    return static_cast<std::uint32_t>(value);
}

// One cached read per process: this is the only place the environment is consulted, so a running
// serve cannot observe a half-changed arm.
[[nodiscard]] inline const LookupRuntimeConfig& lookup_runtime_config() {
    static const LookupRuntimeConfig config = [] {
        LookupRuntimeConfig out;
        out.enabled    = env_flag("NINFER_TP4_LOOKUP", false);
        out.min_match  = std::clamp(env_uint("NINFER_TP4_LOOKUP_MIN", 3), 2U, kMaxSuffix);
        out.min_run    = std::min(env_uint("NINFER_TP4_LOOKUP_MIN_RUN", 2), kBackwardCap);
        out.max_drafts = env_uint("NINFER_TP4_LOOKUP_MAX", 0);
        if (const char* prefix = std::getenv("NINFER_TP4_LOOKUP_DUMP");
            prefix != nullptr && *prefix != '\0') {
            out.dump_prefix = prefix;
        }
        return out;
    }();
    return config;
}

// Fixed-size 4-way bucket table over ledger positions.  No allocation after `ensure`, no churn,
// and a bounded memory footprint: positions are uint32, so a 2^19-slot table is 8 MiB.
class LookupDraftIndex {
public:
    void ensure(std::uint32_t capacity, std::uint32_t min_match) {
        const std::uint32_t wanted_match = std::clamp(min_match, 2U, kMaxSuffix);
        std::uint32_t wanted_slots       = kMinimumSlots;
        const std::uint64_t ceiling      = static_cast<std::uint64_t>(capacity) + 1ULL;
        while (wanted_slots < ceiling && wanted_slots < kMaximumSlots) { wanted_slots <<= 1; }
        if (min_match_ == wanted_match && !table_.empty() &&
            table_.size() == static_cast<std::size_t>(wanted_slots)) {
            return;
        }
        min_match_ = wanted_match;
        table_.clear();
        table_.resize(wanted_slots);
        mask_ = wanted_slots - 1U;
        reset_positions();
    }

    // Bring the index in line with `tokens`: append the windows that became complete, and rebuild
    // when the ledger shrank (prefix reuse truncates it, clear_lane empties it).
    void sync(std::span<const TokenId> tokens) {
        if (table_.empty()) { return; }
        const std::size_t length = tokens.size();
        if (length < indexed_length_) { reset_positions(); }
        indexed_length_ = length;
        if (min_match_ == 0 || length < min_match_) { return; }
        // A window starting at p occupies [p, p + m) and is indexed only while its END is at or
        // before length - 1, i.e. p + m <= length - 1.  That exactly excludes the current suffix
        // window (p == length - m) without ever having to remove an entry later: the window that
        // is the suffix now becomes indexable one token later.
        while (static_cast<std::size_t>(next_position_) + min_match_ < length) {
            insert(tokens.data() + next_position_, next_position_);
            ++next_position_;
        }
    }

    // Propose up to `k` tokens that followed the most convincing earlier copy of the current
    // suffix.  Returns 0 when there is no usable candidate (`k == 0` included).
    [[nodiscard]] std::uint32_t propose(std::span<const TokenId> tokens, std::uint32_t k,
                                        std::uint32_t min_run, TokenId* out) const {
        if (table_.empty() || min_match_ == 0 || k == 0) { return 0; }
        const std::size_t length = tokens.size();
        if (length < static_cast<std::size_t>(min_match_) + 1U) { return 0; }
        const std::size_t suffix = length - min_match_;
        const Bucket& bucket     = table_[bucket_of(tokens.data() + suffix)];

        std::size_t best_position = length;  // sentinel: no candidate yet
        std::uint32_t best_run    = 0;
        for (std::uint32_t way = 0; way < bucket.count; ++way) {
            const std::uint32_t position = bucket.positions[way];
            if (position >= suffix) { continue; }          // stale after a truncation, or the self-hit
            if (!matches(tokens, position, suffix)) { continue; }
            const std::uint32_t run = backward_run(tokens, position, suffix);
            if (best_position == length || run > best_run ||
                (run == best_run && position > best_position)) {
                best_position = position;
                best_run      = run;
            }
        }
        if (best_position == length || best_run < min_run) { return 0; }

        const std::size_t start = best_position + min_match_;
        const std::size_t available = length - start;       // >= 1: best_position < suffix
        const std::uint32_t count =
            static_cast<std::uint32_t>(std::min<std::size_t>(k, available));
        for (std::uint32_t i = 0; i < count; ++i) { out[i] = tokens[start + i]; }
        return count;
    }

    [[nodiscard]] std::uint32_t min_match() const noexcept { return min_match_; }
    [[nodiscard]] std::size_t indexed_windows() const noexcept { return next_position_; }

    // Drop every indexed position. `sync` also rebuilds on its own when the ledger SHRINKS, but a
    // ledger REPLACED by a longer-or-equal one (the next request's prompt: program_impl.h:866
    // `ledger.assign`) keeps its length, so it would keep both the stale windows and the stale
    // `next_position_` -- and the new prompt's first `next_position_` tokens would never be
    // indexed. Measured cost of that defect: 1 non-fire in 3069 rounds (crosscheck_round.py,
    // 2026-10-14), i.e. the copy source has to sit near the START of the new prompt.
    void reset() { reset_positions(); }

private:
    struct Bucket {
        std::uint32_t positions[kWaysPerBucket]{};
        std::uint32_t count = 0;   // 0 = empty; filled slots are [0, count)
        std::uint32_t next  = 0;   // victim for the next overflow
    };

    void reset_positions() {
        for (Bucket& bucket : table_) { bucket = Bucket{}; }
        next_position_  = 0;
        indexed_length_ = 0;
    }

    void insert(const TokenId* window, std::uint32_t position) {
        Bucket& bucket = table_[hash_window(window) & mask_];
        if (bucket.count < kWaysPerBucket) {
            bucket.positions[bucket.count++] = position;
            return;
        }
        bucket.positions[bucket.next] = position;          // recent-wins eviction
        bucket.next                   = (bucket.next + 1U) % kWaysPerBucket;
    }

    [[nodiscard]] std::uint64_t bucket_of(const TokenId* window) const noexcept {
        return hash_window(window) & mask_;
    }

    [[nodiscard]] std::uint64_t hash_window(const TokenId* window) const noexcept {
        std::uint64_t hash = 1469598103934665603ULL ^ static_cast<std::uint64_t>(min_match_);
        for (std::uint32_t i = 0; i < min_match_; ++i) {
            hash ^= static_cast<std::uint32_t>(window[i]) + 0x9e3779b9U;
            hash *= 1099511628211ULL;
        }
        return hash ^ (hash >> 29);
    }

    [[nodiscard]] bool matches(std::span<const TokenId> tokens, std::uint32_t position,
                               std::size_t suffix) const noexcept {
        for (std::uint32_t i = 0; i < min_match_; ++i) {
            if (tokens[position + i] != tokens[suffix + i]) { return false; }
        }
        return true;
    }

    // How many tokens BEFORE the candidate also agree, capped.  A longer agreement is the better
    // bet, which is what keeps a common bigram from outvoting a full repeat of a long span.
    [[nodiscard]] static std::uint32_t backward_run(std::span<const TokenId> tokens,
                                                    std::uint32_t position,
                                                    std::size_t suffix) noexcept {
        std::uint32_t run = 0;
        while (run < kBackwardCap && run < position && run < suffix &&
               tokens[position - 1U - run] == tokens[suffix - 1U - run]) {
            ++run;
        }
        return run;
    }

    std::vector<Bucket> table_;
    std::uint32_t mask_             = 0;
    std::uint32_t min_match_        = 0;
    std::uint32_t next_position_    = 0;
    std::size_t indexed_length_     = 0;
};

// ---------------- diagnostics (debug-only; nothing is opened unless DUMP names a prefix) ---------

class LookupDiagnostics {
public:
    static LookupDiagnostics& instance() {
        static LookupDiagnostics diagnostics;
        return diagnostics;
    }

    void open(const std::string& prefix) {
        if (prefix.empty() || round_file_ != nullptr) { return; }
        round_file_ = std::fopen((prefix + ".round").c_str(), "a");
        request_file_ = std::fopen((prefix + ".req").c_str(), "a");
        if (round_file_ != nullptr) { std::setvbuf(round_file_, nullptr, _IOLBF, 0); }
        if (request_file_ != nullptr) { std::setvbuf(request_file_, nullptr, _IOLBF, 0); }
    }

    void round(std::uint32_t lane, std::uint32_t committed, std::uint32_t fired,
               std::uint32_t proposed, std::uint32_t horizon) {
        if (round_file_ == nullptr) { return; }
        std::fprintf(round_file_, "ROUND %u %llu %u %u %u %u\n", lane,
                     static_cast<unsigned long long>(round_sequence_++), committed, fired, proposed,
                     horizon);
    }

    void request(std::uint32_t lane, std::uint32_t prompt_tokens, std::span<const TokenId> tokens) {
        if (request_file_ == nullptr) { return; }
        std::fprintf(request_file_, "REQ %u %u %zu", lane, prompt_tokens, tokens.size());
        for (const TokenId token : tokens) { std::fprintf(request_file_, " %d", token); }
        std::fputc('\n', request_file_);
    }

private:
    LookupDiagnostics() = default;
    std::FILE* round_file_   = nullptr;
    std::FILE* request_file_ = nullptr;
    std::uint64_t round_sequence_ = 0;
};

// ---------------- the single call site in the decode round tail --------------------------------

// Why the dump is gated separately: a baseline boot (LOOKUP=0) still has a ledger, and its real
// token stream is what the offline replay needs to predict the arm WITHOUT a serve restart. So
// NINFER_TP4_LOOKUP_DUMP works while the feature itself is off.
inline void apply_lookup_drafts(LookupDraftIndex& index, const LookupRuntimeConfig& config,
                                std::uint32_t capacity, std::span<const TokenId> ledger,
                                std::uint32_t committed, bool terminal, std::uint32_t lane,
                                std::uint32_t draft_window, TokenId* drafts,
                                std::uint32_t& count) {
    if (!config.enabled) {
        if (!config.dump_prefix.empty()) { LookupDiagnostics::instance().open(config.dump_prefix); }
        return;
    }
    LookupDiagnostics& diagnostics = LookupDiagnostics::instance();
    diagnostics.open(config.dump_prefix);
    index.ensure(capacity, config.min_match);
    index.sync(ledger);
    if (terminal) {
        diagnostics.round(lane, committed, 0, 0, index.indexed_windows() > 0 ? 1U : 0U);
        return;
    }
    std::uint32_t budget = config.max_drafts == 0 ? draft_window : config.max_drafts;
    budget               = std::min(budget, draft_window);
    TokenId proposal[kMaxProposal] = {};
    const std::uint32_t found      = index.propose(ledger, budget, config.min_run, proposal);
    if (found > 0) {
        for (std::uint32_t step = 0; step < found; ++step) { drafts[step] = proposal[step]; }
        count = found;
    }
    diagnostics.round(lane, committed, found > 0 ? 1U : 0U, found, 0U);
}

// Per-request diagnostic: the FINAL ledger with its prompt boundary in front. The offline replay
// (`lookup_sim.py` in ~/bench-results/2026-10-14/lookup_spec/) rebuilds the exact rounds the
// engine would run on this text, which is how the arm is sized before it is worth a serve restart.
inline void dump_request_ledger(const LookupRuntimeConfig& config, std::uint32_t lane,
                                std::uint32_t prompt_tokens, std::span<const TokenId> tokens) {
    if (config.dump_prefix.empty()) { return; }
    LookupDiagnostics& diagnostics = LookupDiagnostics::instance();
    diagnostics.open(config.dump_prefix);
    diagnostics.request(lane, prompt_tokens, tokens);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::lookup
