// Direct-call probe for the op-level chunked GDN path.
//
// The standalone probe (gdn_chunked_v3_probe.cu) validates the algorithm with its own layout and
// pre-normalised data. The in-tree path differs in three ways: q/k/v/out are [T][heads][D]
// token-major tensors, GQA maps value head -> qk head, and it performs the L2 normalisation
// itself. This probe calls launch_chunked_volta with the real tensor layouts so the dump can be
// checked by the same Python reference, isolating "kernel port" from "dispatch/workspace".

#include "ninfer/ops/gated_delta_net.h"

#include "ninfer_bench_common.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "ops/linear_attention/gated_delta_net/chunked_volta.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

namespace {

float frand(std::uint32_t& s) {
    s = s * 1664525u + 1013904223u;
    return (static_cast<float>(s >> 8) / 16777216.0f) * 2.0f - 1.0f;
}

} // namespace

int main(int argc, char** argv) {
    const int hv     = argc > 1 ? std::atoi(argv[1]) : 24;
    const int hq     = argc > 2 ? std::atoi(argv[2]) : 8;
    const int tokens = argc > 3 ? std::atoi(argv[3]) : 3072;
    const char* dump = argc > 4 ? argv[4] : "/tmp/gdn_op_chunk.bin";
    const bool pre_normalized = argc > 5 && std::atoi(argv[5]) != 0;  // host-side normalise
    const int D      = 128;

    // op layout: q/k [D][Hq][T], v/out [D][Hv][T], g/beta [Hv][T], state [D][D][Hv]
    std::vector<__nv_bfloat16> hq_t(static_cast<std::size_t>(hq) * tokens * D);
    std::vector<__nv_bfloat16> hk_t(hq_t.size());
    std::vector<__nv_bfloat16> hv_t(static_cast<std::size_t>(hv) * tokens * D);
    std::vector<float> hg(static_cast<std::size_t>(hv) * tokens);
    std::vector<float> hbeta(hg.size());
    std::vector<float> hstate(static_cast<std::size_t>(hv) * D * D);

    std::uint32_t seed = 4242u;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < hq; ++h) {
            for (int d = 0; d < D; ++d) {
                const std::size_t idx = (static_cast<std::size_t>(t) * hq + h) * D + d;
                hq_t[idx]             = __float2bfloat16(frand(seed));
                hk_t[idx]             = __float2bfloat16(frand(seed));
            }
        }
        for (int h = 0; h < hv; ++h) {
            for (int d = 0; d < D; ++d) {
                hv_t[(static_cast<std::size_t>(t) * hv + h) * D + d] =
                    __float2bfloat16(0.5f * frand(seed));
            }
            hg[static_cast<std::size_t>(t) * hv + h]         = -0.05f;
            hbeta[static_cast<std::size_t>(t) * hv + h]      = 0.6f;
        }
    }
    for (int h = 0; h < hv; ++h) {
        for (int v = 0; v < D; ++v) {
            for (int k = 0; k < D; ++k) {
                hstate[(static_cast<std::size_t>(h) * D + v) * D + k] = 0.2f * frand(seed);
            }
        }
    }

    if (pre_normalized) {
        for (int t = 0; t < tokens; ++t) {
            for (int h = 0; h < hq; ++h) {
                float nq = 0.0f, nk = 0.0f;
                for (int d = 0; d < D; ++d) {
                    const std::size_t idx = (static_cast<std::size_t>(t) * hq + h) * D + d;
                    const float qv = __bfloat162float(hq_t[idx]);
                    const float kv = __bfloat162float(hk_t[idx]);
                    nq += qv * qv; nk += kv * kv;
                }
                const float iq = 1.0f / std::sqrt(nq + 1.0e-6f);
                const float ik = 1.0f / std::sqrt(nk + 1.0e-6f);
                for (int d = 0; d < D; ++d) {
                    const std::size_t idx = (static_cast<std::size_t>(t) * hq + h) * D + d;
                    hq_t[idx] = __float2bfloat16(__bfloat162float(hq_t[idx]) * iq);
                    hk_t[idx] = __float2bfloat16(__bfloat162float(hk_t[idx]) * ik);
                }
            }
        }
    }
    ninfer::DeviceBuffer dq(hq_t.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer dk(hk_t.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer dv(hv_t.size() * sizeof(__nv_bfloat16));
    ninfer::DeviceBuffer dg(hg.size() * sizeof(float));
    ninfer::DeviceBuffer dbeta(hbeta.size() * sizeof(float));
    ninfer::DeviceBuffer dstate(hstate.size() * sizeof(float));
    ninfer::DeviceBuffer dout(hv_t.size() * sizeof(__nv_bfloat16));

    CUDA_CHECK(cudaMemcpy(dq.p, hq_t.data(), dq.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dk.p, hk_t.data(), dk.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv.p, hv_t.data(), dv.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dg.p, hg.data(), dg.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dbeta.p, hbeta.data(), dbeta.bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dstate.p, hstate.data(), dstate.bytes, cudaMemcpyHostToDevice));

    // The op's tensors describe [dim][heads][tokens]; the kernels index token-major inside a token.
    ninfer::Tensor tq(dq.p, ninfer::DType::BF16, {D, hq, tokens});
    ninfer::Tensor tk(dk.p, ninfer::DType::BF16, {D, hq, tokens});
    ninfer::Tensor tv(dv.p, ninfer::DType::BF16, {D, hv, tokens});
    ninfer::Tensor tg(dg.p, ninfer::DType::FP32, {hv, tokens, 1, 1});
    ninfer::Tensor tbeta(dbeta.p, ninfer::DType::FP32, {hv, tokens, 1, 1});
    ninfer::Tensor tstate(dstate.p, ninfer::DType::FP32, {D, D, hv, 1});
    ninfer::Tensor tout(dout.p, ninfer::DType::BF16, {D, hv, tokens});

    std::size_t bytes = 0;
    {
        ninfer::WorkspaceArena probe_ws(1);
        struct Counter {
            std::size_t total = 0;
            ninfer::DeviceSpan alloc_bytes(std::size_t n, std::size_t = 256) {
                total += n;
                return {nullptr, n};
            }
        } counter;
        ninfer::ops::detail::gated_delta_net::chunked_volta_allocate(counter, hv, tokens);
        bytes = counter.total + 4096;
    }
    ninfer::WorkspaceArena ws(bytes);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    ninfer::ops::detail::gated_delta_net::launch_chunked_volta(
        tq, tk, tv, tg, tbeta, 1.0f / std::sqrt(static_cast<float>(D)), /*normalize_qk=*/!pre_normalized, ws,
        tstate, tstate, tout, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<__nv_bfloat16> oout(hv_t.size());
    std::vector<float> ostate(hstate.size());
    CUDA_CHECK(cudaMemcpy(oout.data(), dout.p, dout.bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ostate.data(), dstate.p, dstate.bytes, cudaMemcpyDeviceToHost));
    std::printf("probe out ptr %p (tout.data %p) bytes %zu\n", dout.p, tout.data, dout.bytes);
    std::printf("out[0..3] = %.6f %.6f %.6f %.6f  |out| = %.4f\n",
                __bfloat162float(oout[0]), __bfloat162float(oout[1]),
                __bfloat162float(oout[2]), __bfloat162float(oout[3]),
                [&] { double n = 0; for (auto x : oout) { const double v = __bfloat162float(x); n += v * v; } return std::sqrt(n); }());
    std::printf("ostate[0] = %.3f (888 => kernel wrote state; -1 => not)\n", ostate[0]);
    std::printf("state changed: %s\n",
                (ostate == hstate) ? "no (!!)" : "yes");

    // Dump in the standalone probe's format so gdn_chunk_v2_validate.py applies unchanged.
    std::vector<__nv_bfloat16> hq_t_n = hq_t;
    std::vector<__nv_bfloat16> hk_t_n = hk_t;
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < hq; ++h) {
            float nq = 0.0f, nk = 0.0f;
            for (int d = 0; d < D; ++d) {
                const std::size_t idx = (static_cast<std::size_t>(t) * hq + h) * D + d;
                const float qv = __bfloat162float(hq_t[idx]);
                const float kv = __bfloat162float(hk_t[idx]);
                nq += qv * qv; nk += kv * kv;
            }
            const float iq = 1.0f / std::sqrt(nq + 1.0e-6f);
            const float ik = 1.0f / std::sqrt(nk + 1.0e-6f);
            for (int d = 0; d < D; ++d) {
                const std::size_t idx = (static_cast<std::size_t>(t) * hq + h) * D + d;
                hq_t_n[idx] = __float2bfloat16(__bfloat162float(hq_t[idx]) * iq);
                hk_t_n[idx] = __float2bfloat16(__bfloat162float(hk_t[idx]) * ik);
            }
        }
    }
    std::ofstream f(dump, std::ios::binary);
    const std::int32_t meta[3] = {hv, tokens, 32};
    f.write(reinterpret_cast<const char*>(meta), sizeof(meta));
    // q and k expanded per value head with the GQA mapping the kernels use
    std::vector<__half> qexp(static_cast<std::size_t>(hv) * tokens * D);
    std::vector<__half> kexp(qexp.size());
    std::vector<__half> vexp(hv_t.size());
    std::vector<__half> oexp(hv_t.size());
    for (int h = 0; h < hv; ++h) {
        const int hk = h % hq;
        for (int t = 0; t < tokens; ++t) {
            for (int d = 0; d < D; ++d) {
                const std::size_t src = (static_cast<std::size_t>(t) * hq + hk) * D + d;
                const std::size_t dst = ((static_cast<std::size_t>(h) * tokens + t)) * D + d;
                qexp[dst] = __float2half(__bfloat162float(hq_t[src]));
                kexp[dst] = __float2half(__bfloat162float(hk_t[src]));
                // the Python reference reads every half buffer as FP16, so convert on dump
                vexp[dst] = __float2half(__bfloat162float(hv_t[(static_cast<std::size_t>(t) * hv + h) * D + d]));
                oexp[dst] = __float2half(__bfloat162float(oout[(static_cast<std::size_t>(t) * hv + h) * D + d]));
            }
        }
    }
    // validate.py expects q/k as [T][D] (head-agnostic): write head 0's copy
    std::vector<__half> q0(static_cast<std::size_t>(tokens) * D);
    std::vector<__half> k0(q0.size());
    // dump the *normalised* rows: the kernel normalises internally (rsqrtf(sum + 1e-6)), and the
    // Python reference must see exactly those values.
    for (int t = 0; t < tokens; ++t) {
        float nq = 0.0f, nk = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float qv = __bfloat162float(hq_t[(static_cast<std::size_t>(t) * hq) * D + d]);
            const float kv = __bfloat162float(hk_t[(static_cast<std::size_t>(t) * hq) * D + d]);
            nq += qv * qv;
            nk += kv * kv;
        }
        const float iq = 1.0f / std::sqrt(nq + 1.0e-6f);
        const float ik = 1.0f / std::sqrt(nk + 1.0e-6f);
        for (int d = 0; d < D; ++d) {
            const std::size_t src = (static_cast<std::size_t>(t) * hq) * D + d;
            q0[static_cast<std::size_t>(t) * D + d] =
                __float2half(__bfloat162float(hq_t[src]) * iq);
            k0[static_cast<std::size_t>(t) * D + d] =
                __float2half(__bfloat162float(hk_t[src]) * ik);
        }
    }
    f.write(reinterpret_cast<const char*>(q0.data()), q0.size() * sizeof(__nv_bfloat16));
    f.write(reinterpret_cast<const char*>(k0.data()), k0.size() * sizeof(__nv_bfloat16));
    f.write(reinterpret_cast<const char*>(vexp.data()), vexp.size() * sizeof(__nv_bfloat16));
    f.write(reinterpret_cast<const char*>(hg.data()), hg.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(hbeta.data()), hbeta.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(hstate.data()), hstate.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(ostate.data()), ostate.size() * sizeof(float));
    f.write(reinterpret_cast<const char*>(oexp.data()), oexp.size() * sizeof(__nv_bfloat16));
    f.close();
    std::printf("dump: %s (hv=%d hq=%d T=%d)\n", dump, hv, hq, tokens);
    return 0;
}
