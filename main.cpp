// ============================================================================
// File: main.cpp
// ============================================================================

#include <chrono>
#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "hypergraph.h"
#include "freq_model.h"
#include "freq_code.h"
#include "encode.h"
#include "decode.h"
#include "BFS.h"
#include "computeKCore.h"
#include "pagerank.h"

using namespace std;

// -----------------------------------------------------------------------------
// Memory footprint estimation helpers
// -----------------------------------------------------------------------------
static size_t footprint(int n, long bitsH, long bitsL) {
    long bH = (bitsH + 7) / 8, bL = (bitsL + 7) / 8;
    return 2 * sizeof(int*)
         + sizeof(uint16_t*)
         + 2 * sizeof(uint8_t*)
         + sizeof(int) * n
         + sizeof(int) * n
         + bH * sizeof(uint8_t)
         + bL * sizeof(uint8_t)
         + sizeof(uint16_t) * n
         + 2 * sizeof(uint64_t) * ((n + 1024 - 1) / 1024);
}

static size_t footprint_one(int n, long bitsH, long bitsL) {
    long bH = (bitsH + 7) / 8, bL = (bitsL + 7) / 8;
    return 2 * sizeof(int*)
         + sizeof(uint16_t*)
         + 2 * sizeof(uint8_t*)
         + sizeof(uint16_t) * n
         + sizeof(int) * n
         + bH * sizeof(uint8_t)
         + bL * sizeof(uint8_t)
         + sizeof(uint16_t) * n
         + 2 * sizeof(uint64_t) * ((n + 1024 - 1) / 1024);
}

static size_t footprint_both(int n, long bitsH, long bitsL) {
    long bH = (bitsH + 7) / 8, bL = (bitsL + 7) / 8;
    return 2 * sizeof(uint16_t*)
         + sizeof(uint16_t*)
         + 2 * sizeof(uint8_t*)
         + sizeof(uint16_t) * n
         + sizeof(uint16_t) * n
         + bH * sizeof(uint8_t)
         + bL * sizeof(uint8_t)
         + sizeof(uint16_t) * n
         + 2 * sizeof(uint64_t) * ((n + 1024 - 1) / 1024);
}

static inline size_t compute_raw_side_bytes(
    bool encodeH,
    int nv, int nh,
    const int* degreeV, const int* degreeH
) {
    const int   n   = encodeH ? nh : nv;
    const int*  deg = encodeH ? degreeH : degreeV;

    uint64_t m = 0;
    for (int i = 0; i < n; ++i) m += (uint32_t)deg[i];

    return (size_t)n * sizeof(int)
         + (size_t)m * sizeof(int);
}

// -----------------------------------------------------------------------------
// RSS helper: measure current resident set size in KB
// -----------------------------------------------------------------------------
static size_t getCurrentRSSKB() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;

    char line[256];
    size_t rss_kb = 0;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%zu", &rss_kb);
            break;
        }
    }
    fclose(f);
    return rss_kb;
}

// -----------------------------------------------------------------------------
// Timing stats
// -----------------------------------------------------------------------------
struct MetricStats {
    long long total_ns = 0;
    long long max_ns = 0;
    long long count = 0;

    void add(long long x) {
        total_ns += x;
        max_ns = std::max(max_ns, x);
        count++;
    }

    double avg_us() const {
        return count ? (double)total_ns / count / 1000.0 : 0.0;
    }

    double max_us() const {
        return (double)max_ns / 1000.0;
    }
};

struct BlockTimingStats {
    MetricStats setup;
    MetricStats single_decode;
    MetricStats full_block_decode;

    MetricStats bfs_app;
    MetricStats bfs_e2e;

    MetricStats pr_app;
    MetricStats pr_e2e;

    MetricStats kcore_app;
    MetricStats kcore_e2e;

    long long blocks_measured = 0;
    long long entries_measured = 0;
};


int main(int argc, char** argv) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0]
             << " <pct> <input.hyper> <k_thresh> <lp_iters> <pr_iters>\n";
        return 1;
    }

    // -------------------------------------------------------------------------
    // Parse params
    // -------------------------------------------------------------------------
    double pct = atof(argv[1]) * 0.01;
    const char* fname = argv[2];
    int k_thresh = atoi(argv[3]);
    int lp_iters = atoi(argv[4]);
    int pr_iters = atoi(argv[5]);

    Hypergraph* G = readHypergraph(fname);
    int nv = G->nv, nh = G->nh;
    bool encodeH = nv <= nh;

    // -------------------------------------------------------------------------
    // Encoding
    // -------------------------------------------------------------------------
    cout << "Encoding" << endl;

    int fbV = 0, fbH = 0;
    HuffmanNode *tV = nullptr, *tH = nullptr;
    unordered_map<int, string> cV, cH;
    uint16_t *bitCount = nullptr;
    uint16_t *huffCount = nullptr;
    uint8_t *hi = nullptr, *lo = nullptr;
    int bitsH = 0, bitsL = 0;

    uint64_t *blockHi = nullptr, *blockLo = nullptr;
    int nBlocks = 0;
    const int BLOCK = 1024;

    auto t_encode0 = chrono::high_resolution_clock::now();

    if (encodeH) {
        tV = buildTree(G->degreeV, nv, pct, fbV);
        genCodes(tV, cV);
        tie(bitsH, bitsL) = encodeSide(
            nh, G->degreeH, G->edgesH, cV, fbV,
            huffCount, bitCount, hi, lo,
            BLOCK, blockHi, blockLo, &nBlocks
        );
    } else {
        tH = buildTree(G->degreeH, nh, pct, fbH);
        genCodes(tH, cH);
        tie(bitsH, bitsL) = encodeSide(
            nv, G->degreeV, G->edgesV, cH, fbH,
            huffCount, bitCount, hi, lo,
            BLOCK, blockHi, blockLo, &nBlocks
        );
    }

    auto t_encode1 = chrono::high_resolution_clock::now();
    cout << "Encoding Time: "
         << chrono::duration<double>(t_encode1 - t_encode0).count()
         << " s\n";

    // -------------------------------------------------------------------------
    // Footprint estimate
    // -------------------------------------------------------------------------
    size_t treeSize = encodeH ? treeMem(tV) : treeMem(tH);
    int n = encodeH ? nh : nv;

    int maxDeg = 0, maxBC = 0;
    for (int i = 0; i < n; ++i) {
        maxDeg = max(maxDeg, encodeH ? G->degreeH[i] : G->degreeV[i]);
        maxBC  = max(maxBC, (int)bitCount[i]);
    }

    size_t contentSize = 0;
    if (maxDeg < (1 << 16) && maxBC < (1 << 16)) {
        contentSize = footprint_both(n, bitsH, bitsL);
    } else if (maxDeg < (1 << 16) || maxBC < (1 << 16)) {
        contentSize = footprint_one(n, bitsH, bitsL);
    } else {
        contentSize = footprint(n, bitsH, bitsL);
    }

    cout << "Content size: " << (contentSize / 1024) << " KB\n";
    cout << "Tree    size: " << (treeSize / 1024) << " KB\n";
    cout << "Total   size: " << ((contentSize + treeSize) / 1024) << " KB\n";

    // -------------------------------------------------------------------------
    // Helpers for blocked microbenchmarks
    // -------------------------------------------------------------------------
    using clock_t = std::chrono::high_resolution_clock;
    using ns = std::chrono::nanoseconds;

    auto decode_single_entry_from_slices =
        [&](const int*       deg_blk,
            const uint16_t*  bit_blk,
            const uint16_t*  huff_blk,
            const uint8_t*   hiSlice,
            const uint8_t*   loSlice,
            uint32_t         hiBitOffset,
            uint32_t         loBitOffset,
            HuffmanNode*     treeOpp,
            int              fallbackBitsOpp) -> std::vector<int>
    {
        std::vector<int> out;
        const int d = deg_blk[0];
        if (d <= 0) return out;

        out.reserve(d);

        const int h = huff_blk[0];
        const int hb = bit_blk[0];

        uint64_t ptrHi_bits = hiBitOffset;
        uint64_t ptrLo_bits = loBitOffset;

        const uint64_t hiStart = ptrHi_bits;
        int usedH = 0;
        int usedBit = 0;

        while (usedH < h && usedBit < hb) {
            HuffmanNode* cur = treeOpp;
            do {
                bool bit = (hiSlice[ptrHi_bits >> 3] >> (ptrHi_bits & 7)) & 1;
                cur = bit ? cur->r : cur->l;
                ++ptrHi_bits;
                ++usedBit;
            } while (cur->l || cur->r);
            out.push_back(cur->v);
            ++usedH;
        }

        const uint64_t shouldBe = hiStart + (uint64_t)hb;
        if (ptrHi_bits < shouldBe) ptrHi_bits = shouldBe;

        for (int t = usedH; t < d; ++t) {
            long tmp = static_cast<long>(ptrLo_bits);
            int x = readBits(loSlice, tmp, fallbackBitsOpp);
            ptrLo_bits = static_cast<uint64_t>(tmp);
            out.push_back(x);
        }

        return out;
    };

    auto decode_all_entries_in_block =
        [&](const std::vector<int>& deg_blk,
            const std::vector<uint16_t>& bit_blk,
            const std::vector<uint16_t>& huff_blk,
            const std::vector<uint8_t>& hiSlice,
            const std::vector<uint8_t>& loSlice,
            uint32_t hiBitOffset,
            uint32_t loBitOffset,
            HuffmanNode* treeOpp,
            int fallbackBitsOpp,
            std::vector<std::vector<int>>& decoded_block) -> void
    {
        const int L = (int)deg_blk.size();
        decoded_block.clear();
        decoded_block.resize(L);

        std::vector<uint64_t> offHiBits(L, 0), offLoBits(L, 0);
        uint64_t accH = 0, accL = 0;
        for (int k = 0; k < L; ++k) {
            offHiBits[k] = accH;
            offLoBits[k] = accL;
            accH += (uint64_t)bit_blk[k];
            accL += (uint64_t)(deg_blk[k] - huff_blk[k]) * (uint64_t)fallbackBitsOpp;
        }

        for (int k = 0; k < L; ++k) {
            decoded_block[k] = decode_single_entry_from_slices(
                &deg_blk[k],
                &bit_blk[k],
                &huff_blk[k],
                hiSlice.data(),
                loSlice.data(),
                hiBitOffset + (uint32_t)offHiBits[k],
                loBitOffset + (uint32_t)offLoBits[k],
                treeOpp,
                fallbackBitsOpp
            );
        }
    };

    // -------------------------------------------------------------------------
    // Microbenchmark: blocked random-access decode latency + app time
    // For each sampled block, report:
    //   decode only
    //   app only
    //   decode + app
    // for BFS, PageRank, and k-core
    // -------------------------------------------------------------------------
    {
        const int NUM_QUERIES = std::min(n, 1000);
        std::vector<int> queries(NUM_QUERIES);
        for (int i = 0; i < NUM_QUERIES; ++i) {
            queries[i] = (long long)i * n / NUM_QUERIES;
        }

        const int* degSide = encodeH ? G->degreeH : G->degreeV;
        HuffmanNode* treeOpp = encodeH ? tV : tH;
        int fallbackBitsOpp = encodeH ? fbV : fbH;

        BlockTimingStats stats;
        uint64_t app_checksum = 0;

        for (int u : queries) {
            const int blk = u / BLOCK;
            const int i0 = blk * BLOCK;
            const int i1 = std::min(n, i0 + BLOCK);
            const int L = i1 - i0;

            std::vector<int>       deg_blk;
            std::vector<uint16_t>  bit_blk, huff_blk;
            std::vector<uint8_t>   hiSlice, loSlice;
            uint32_t               hiBitOffset = 0, loBitOffset = 0;

            // -------------------------------------------------------------
            // 1. block setup
            // -------------------------------------------------------------
            auto t0 = clock_t::now();

            extract_block_slices(
                i0, i1,
                BLOCK,
                degSide,
                bitCount,
                huffCount,
                hi, lo,
                blockHi, blockLo,
                fallbackBitsOpp,
                deg_blk, bit_blk, huff_blk,
                hiSlice, loSlice,
                hiBitOffset, loBitOffset
            );

            std::vector<uint64_t> offHiBits(L, 0), offLoBits(L, 0);
            uint64_t accH = 0, accL = 0;
            for (int k = 0; k < L; ++k) {
                offHiBits[k] = accH;
                offLoBits[k] = accL;
                accH += (uint64_t)bit_blk[k];
                accL += (uint64_t)(deg_blk[k] - huff_blk[k]) * (uint64_t)fallbackBitsOpp;
            }

            auto t1 = clock_t::now();
            stats.setup.add(std::chrono::duration_cast<ns>(t1 - t0).count());

            // -------------------------------------------------------------
            // 2. single-entry local decode
            // -------------------------------------------------------------
            const int k = u - i0;
            const uint32_t hiOff = hiBitOffset + (uint32_t)offHiBits[k];
            const uint32_t loOff = loBitOffset + (uint32_t)offLoBits[k];

            auto t2 = clock_t::now();

            auto nbrs = decode_single_entry_from_slices(
                &deg_blk[k],
                &bit_blk[k],
                &huff_blk[k],
                hiSlice.data(),
                loSlice.data(),
                hiOff,
                loOff,
                treeOpp,
                fallbackBitsOpp
            );

            auto t3 = clock_t::now();
            stats.single_decode.add(std::chrono::duration_cast<ns>(t3 - t2).count());
            (void)nbrs;

            // -------------------------------------------------------------
            // 3. full block decode once
            // -------------------------------------------------------------
            std::vector<std::vector<int>> decoded_block;

            auto t4 = clock_t::now();

            decode_all_entries_in_block(
                deg_blk, bit_blk, huff_blk,
                hiSlice, loSlice,
                hiBitOffset, loBitOffset,
                treeOpp, fallbackBitsOpp,
                decoded_block
            );

            auto t5 = clock_t::now();
            long long full_block_decode_ns = std::chrono::duration_cast<ns>(t5 - t4).count();
            stats.full_block_decode.add(full_block_decode_ns);

            // -------------------------------------------------------------
            // 4. BFS application only, and decode + BFS
            // -------------------------------------------------------------
            auto t6_bfs = clock_t::now();
            uint64_t bfs_ck = run_block_bfs(decoded_block, i0, u);
            auto t7_bfs = clock_t::now();

            long long bfs_app_ns = std::chrono::duration_cast<ns>(t7_bfs - t6_bfs).count();
            stats.bfs_app.add(bfs_app_ns);
            stats.bfs_e2e.add(full_block_decode_ns + bfs_app_ns);

            // -------------------------------------------------------------
            // 5. PageRank application only, and decode + PageRank
            // -------------------------------------------------------------
            auto t6_pr = clock_t::now();
            double pr_ck = run_block_pagerank_iteration(decoded_block, i0, pr_iters, 0.85);
            auto t7_pr = clock_t::now();

            long long pr_app_ns = std::chrono::duration_cast<ns>(t7_pr - t6_pr).count();
            stats.pr_app.add(pr_app_ns);
            stats.pr_e2e.add(full_block_decode_ns + pr_app_ns);

            // -------------------------------------------------------------
            // 6. k-core application only, and decode + k-core
            // -------------------------------------------------------------
            auto t6_kc = clock_t::now();
            uint64_t kc_ck = run_block_kcore(decoded_block, i0, k_thresh);
            auto t7_kc = clock_t::now();

            long long kcore_app_ns = std::chrono::duration_cast<ns>(t7_kc - t6_kc).count();
            stats.kcore_app.add(kcore_app_ns);
            stats.kcore_e2e.add(full_block_decode_ns + kcore_app_ns);

            app_checksum += bfs_ck + kc_ck + (uint64_t)(pr_ck * 1e9);

            stats.blocks_measured++;
            stats.entries_measured += L;
        }

        cout << "Blocked decode/application microbenchmark over "
             << stats.blocks_measured << " sampled blocks:\n";

        cout << "  Avg block setup time              = "
             << stats.setup.avg_us() << " us\n";
        cout << "  Avg single-entry decode           = "
             << stats.single_decode.avg_us() << " us\n";
        cout << "  Avg full-block decode             = "
             << stats.full_block_decode.avg_us() << " us\n";

        cout << "\n";
        cout << "  BFS block application time        = "
             << stats.bfs_app.avg_us() << " us\n";
        cout << "  BFS decode + application          = "
             << stats.bfs_e2e.avg_us() << " us\n";

        cout << "  PageRank block application time   = "
             << stats.pr_app.avg_us() << " us\n";
        cout << "  PageRank decode + application     = "
             << stats.pr_e2e.avg_us() << " us\n";

        cout << "  K-core block application time     = "
             << stats.kcore_app.avg_us() << " us\n";
        cout << "  K-core decode + application       = "
             << stats.kcore_e2e.avg_us() << " us\n";

        cout << "\n";
        cout << "  Max full-block decode             = "
             << stats.full_block_decode.max_us() << " us\n";
        cout << "  Max BFS application               = "
             << stats.bfs_app.max_us() << " us\n";
        cout << "  Max BFS decode + application      = "
             << stats.bfs_e2e.max_us() << " us\n";
        cout << "  Max PageRank application          = "
             << stats.pr_app.max_us() << " us\n";
        cout << "  Max PageRank decode + application = "
             << stats.pr_e2e.max_us() << " us\n";
        cout << "  Max K-core application            = "
             << stats.kcore_app.max_us() << " us\n";
        cout << "  Max K-core decode + application   = "
             << stats.kcore_e2e.max_us() << " us\n";
    }

    // -------------------------------------------------------------------------
    // Full-decode memory measurement
    // Measures RSS increase caused by fully reconstructing the encoded side
    // -------------------------------------------------------------------------
    {
        const int* degSide = encodeH ? G->degreeH : G->degreeV;
        HuffmanNode* treeOpp = encodeH ? tV : tH;
        int fallbackBitsOpp = encodeH ? fbV : fbH;

        size_t rss_before_kb = getCurrentRSSKB();

        int* deg_dec = new int[n];
        std::copy(degSide, degSide + n, deg_dec);

        uint64_t total_m = 0;
        for (int i = 0; i < n; ++i) total_m += (uint32_t)deg_dec[i];
        int* edges_dec = new int[total_m];

        uint64_t write_pos = 0;

        for (int blk = 0; blk < nBlocks; ++blk) {
            const int i0 = blk * BLOCK;
            const int i1 = std::min(n, i0 + BLOCK);

            std::vector<int>       deg_blk;
            std::vector<uint16_t>  bit_blk, huff_blk;
            std::vector<uint8_t>   hiSlice, loSlice;
            uint32_t               hiBitOffset = 0, loBitOffset = 0;

            extract_block_slices(
                i0, i1,
                BLOCK,
                degSide,
                bitCount,
                huffCount,
                hi, lo,
                blockHi, blockLo,
                fallbackBitsOpp,
                deg_blk, bit_blk, huff_blk,
                hiSlice, loSlice,
                hiBitOffset, loBitOffset
            );

            const int L = i1 - i0;
            std::vector<uint64_t> offHiBits(L, 0), offLoBits(L, 0);
            uint64_t accH = 0, accL = 0;
            for (int k = 0; k < L; ++k) {
                offHiBits[k] = accH;
                offLoBits[k] = accL;
                accH += (uint64_t)bit_blk[k];
                accL += (uint64_t)(deg_blk[k] - huff_blk[k]) * (uint64_t)fallbackBitsOpp;
            }

            for (int k = 0; k < L; ++k) {
                const int d  = deg_blk[k];
                const int h  = huff_blk[k];
                const int hb = bit_blk[k];

                uint64_t ptrHi_bits = hiBitOffset + offHiBits[k];
                uint64_t ptrLo_bits = loBitOffset + offLoBits[k];

                int usedH = 0;
                int usedBit = 0;

                while (usedH < h && usedBit < hb) {
                    HuffmanNode* cur = treeOpp;
                    do {
                        bool bit = (hiSlice[ptrHi_bits >> 3] >> (ptrHi_bits & 7)) & 1;
                        cur = bit ? cur->r : cur->l;
                        ++ptrHi_bits;
                        ++usedBit;
                    } while (cur->l || cur->r);

                    edges_dec[write_pos++] = cur->v;
                    ++usedH;
                }

                ptrHi_bits = hiBitOffset + offHiBits[k] + (uint64_t)hb;

                for (int t = usedH; t < d; ++t) {
                    long tmp = static_cast<long>(ptrLo_bits);
                    int x = readBits(loSlice.data(), tmp, fallbackBitsOpp);
                    ptrLo_bits = static_cast<uint64_t>(tmp);
                    edges_dec[write_pos++] = x;
                }
            }
        }

        size_t rss_after_kb = getCurrentRSSKB();

        cout << "Full decode RSS before: " << rss_before_kb << " KB\n";
        cout << "Full decode RSS after : " << rss_after_kb  << " KB\n";
        cout << "Full decode RSS delta : "
             << (rss_after_kb > rss_before_kb ? rss_after_kb - rss_before_kb : 0)
             << " KB\n";

        delete[] deg_dec;
        delete[] edges_dec;
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    freeTree(tV);
    freeTree(tH);
    delete[] bitCount;
    delete[] huffCount;
    delete[] hi;
    delete[] lo;
    delete[] blockHi;
    delete[] blockLo;
    delete G;

    return 0;
}