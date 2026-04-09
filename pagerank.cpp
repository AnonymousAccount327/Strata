// ============================================================================
// pagerank_rolling_singlefile.cpp (with RSS instrumentation in BYTES)
// - Keeps original function names/signatures
// - Works on the encoded graph (no predecode)
// - Uses rolling bit pointers to avoid O(n) scans per decode
// - Measures RSS per block-load (encoded path) and at entry/exit (raw path)
// ============================================================================

#include <vector>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <sys/resource.h>
#include <unistd.h>
#include <cstdio>

#include "huffman_tree.h"  // defines HuffmanNode { HuffmanNode *l,*r; int v; ... }
#include "decode.h"        // declares readBits(...), extract_block_slices(...)

using namespace std;

// ------------------------ RSS helpers ------------------------
static inline long rss_kb() {
    rusage ru; getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__) && defined(__MACH__)
    return ru.ru_maxrss / 1024;     // macOS: bytes -> KB
#else
    return ru.ru_maxrss;            // Linux: already KB
#endif
}

// Process-wide Resident Set Size in BYTES (page‑granular).
// On Linux: read /proc/self/statm. Elsewhere: fall back to rss_kb()*1024.
static inline size_t rss_bytes() {
#if defined(__linux__)
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return (size_t)rss_kb() * 1024ull;
    unsigned long pages = 0UL, dummy = 0UL;
    if (fscanf(f, "%lu %lu", &dummy, &pages) != 2) { fclose(f); return (size_t)rss_kb() * 1024ull; }
    fclose(f);
    long ps = sysconf(_SC_PAGESIZE); if (ps <= 0) ps = 4096;
    return (size_t)pages * (size_t)ps;
#else
    return (size_t)rss_kb() * 1024ull;
#endif
}

// Adapter: call readBits(const uint8_t*, long&, int) with a uint64_t bit pointer
static inline int readBits64(const uint8_t* buf, uint64_t& bitPtr, int width) {
    long tmp = static_cast<long>(bitPtr);   // readBits expects long&
    int  val = readBits(buf, tmp, width);   // advances tmp internally
    bitPtr = static_cast<uint64_t>(tmp);    // write back to our uint64_t
    return val;
}

// ----------------------------------------------------------------------------
// Progress helper (throttled to print only when % changes)
// ----------------------------------------------------------------------------
struct Progress {
    long long total = 1;
    std::atomic<int> last{-1};
    const char* label = nullptr;

    Progress(long long tot, const char* lbl) : total(std::max(1LL, tot)), label(lbl) {}

    inline void update(long long done) {
        int p = int((100LL * done) / total);
        int prev = last.load(std::memory_order_relaxed);
        if (p > prev) {
            if (last.compare_exchange_strong(prev, p, std::memory_order_relaxed)) {
                // std::cout << "[" << label << "] " << p << "% (" << done << "/" << total << ")\n";
            }
        }
    }
};

// ----------------------------------------------------------------------------
// Raw PageRank on the flat edges[] layout (no decoding).
// Same math as runPageRankOnDemand; uses prefix offsets into edges.
// ----------------------------------------------------------------------------
void runPageRankRaw(
    int n,                    // number of vertices
    const int* deg,           // out-degree per vertex
    const int* edges,         // flat adjacency list, length = sum(deg)
    int iters,
    double alpha /*=0.85*/
) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    if (n <= 0) { std::cout << "PageRank(raw): empty graph\n"; return; }

    // Measure RSS at function entry (before allocations)
    size_t r_entry = rss_bytes();

    // Build prefix offsets to locate neighbors of u in O(1)
    std::vector<uint64_t> off(n + 1, 0);
    for (int i = 0; i < n; ++i) off[i + 1] = off[i] + (uint64_t)deg[i];

    std::vector<double> pr(n, 1.0 / n);
    std::vector<double> next(n, 0.0);

    Progress overall((long long)iters, "PageRank raw overall");

    for (int it = 0; it < iters; ++it) {
        std::fill(next.begin(), next.end(), 0.0);
        double dangling = 0.0;

        Progress iterProg((long long)n, "Raw iteration progress");

        // Push-based update
        for (int u = 0; u < n; ++u) {
            const int du = deg[u];
            if (du == 0) { dangling += pr[u]; iterProg.update(u+1); continue; }

            const double share = pr[u] / du;
            const uint64_t s = off[u];
            const uint64_t e = off[u + 1];

            for (uint64_t p = s; p < e; ++p) {
                int v = edges[p];
                if ((unsigned)v < (unsigned)n) next[v] += share;
            }
            iterProg.update(u+1);
        }

        const double base = (1.0 - alpha) / n;
        const double dang = dangling / n;

        for (int v = 0; v < n; ++v)
            pr[v] = base + alpha * (next[v] + dang);

        overall.update(it+1);
    }

    auto t1 = clock::now();
    size_t r_exit = rss_bytes();
    std::cout << "PageRank (raw edges) finished in "
              << std::chrono::duration<double>(t1 - t0).count() << " s"
              << " | RSS entry = " << r_entry << " bytes"
              << " | RSS exit = " << r_exit << " bytes"
              << " | RSS \xCE\x94 = " << (r_exit > r_entry ? (r_exit - r_entry) : 0) << " bytes\n";
}

static std::vector<int> decode_single_entry_from_slices(
    const int*       deg_blk,        // length >= 1
    const uint16_t*  bit_blk,        // length >= 1
    const uint16_t*  huff_blk,       // length >= 1
    const uint8_t*   hiSlice,
    const uint8_t*   loSlice,
    uint32_t         hiBitOffset,
    uint32_t         loBitOffset,
    HuffmanNode*     treeOpp,
    int              fallbackBitsOpp
) {
    std::vector<int> out;
    const int d  = deg_blk[0];
    if (d <= 0) return out;

    out.reserve(d);

    const int h  = huff_blk[0];
    const int hb = bit_blk[0];

    uint64_t ptrHi_bits = hiBitOffset;  // bit pointer into hiSlice
    uint64_t ptrLo_bits = loBitOffset;  // bit pointer into loSlice

    // ---- Huffman section: decode h symbols within hb bits ----
    const uint64_t hiStart = ptrHi_bits;
    int usedH   = 0;
    int usedBit = 0;
    while (usedH < h && usedBit < hb) {
        HuffmanNode* cur = treeOpp;
        // Walk to leaf
        do {
            bool bit = (hiSlice[ptrHi_bits >> 3] >> (ptrHi_bits & 7)) & 1;
            cur = bit ? cur->r : cur->l;
            ++ptrHi_bits; ++usedBit;
        } while (cur->l || cur->r);
        out.push_back(cur->v);
        ++usedH;
    }
    // Align to hb bits (skip any padding)
    const uint64_t shouldBe = hiStart + (uint64_t)hb;
    if (ptrHi_bits < shouldBe) ptrHi_bits = shouldBe;

    // ---- Fallback section: decode the remaining (d - usedH) fixed-width values ----
    for (int t = usedH; t < d; ++t) {
        int x = readBits64(loSlice, ptrLo_bits, fallbackBitsOpp);
        out.push_back(x);
    }

    return out;
}

// Parallel PageRank that decodes neighbors from *block slices* (cached per thread)
void runPageRankOnDemand_decodeRandom(
    int n, const int* deg, const int* edges,
    const uint16_t* huffCount, const uint16_t* bitCount,
    const uint8_t* hi, const uint8_t* lo,
    HuffmanNode* treeOpp, int fallbackBitsOpp,
    int iters, double alpha,
    const uint64_t* blockHi, const uint64_t* blockLo, int BLOCK
){
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    if (n <= 0) { std::cout << "PageRank: empty graph\n"; return; }
    if (!deg || !huffCount || !bitCount || !hi || !lo || !treeOpp ||
        !blockHi || !blockLo || BLOCK <= 0) {
        std::cerr << "PageRank(decode): missing/invalid inputs\n";
        return;
    }

    const int nBlocks = (n + BLOCK - 1) / BLOCK;

    std::vector<double> pr(n, 1.0 / n);
    std::vector<double> next(n, 0.0);

    // RSS: track peak per-block extraction delta
    size_t peak_block_rss_bytes = 0;

    for (int it = 0; it < iters; ++it) {
        std::fill(next.begin(), next.end(), 0.0);
        double dangling = 0.0;

        // -------------------- PARALLEL OVER BLOCKS --------------------
        #pragma omp parallel
        {
            // Thread-local block cache
            std::vector<int>       t_deg_blk;
            std::vector<uint16_t>  t_bit_blk, t_huff_blk;
            std::vector<uint8_t>   t_hiSlice, t_loSlice;
            uint32_t               t_baseHiOffset = 0, t_baseLoOffset = 0;
            std::vector<uint64_t>  t_offHiBits, t_offLoBits; // intra-block bit offsets
            std::vector<int>       nbrs; nbrs.reserve(128);

            double local_dangling = 0.0;

            #pragma omp for schedule(dynamic,1) reduction(+:dangling)
            for (int b = 0; b < nBlocks; ++b) {
                const int i0 = b * BLOCK;
                const int i1 = std::min(n, i0 + BLOCK);
                if (i0 >= i1) continue;

                // Extract slices ONCE for the whole block, measure RSS around it
                t_deg_blk.clear(); t_bit_blk.clear(); t_huff_blk.clear();
                t_hiSlice.clear();  t_loSlice.clear();
                t_baseHiOffset = t_baseLoOffset = 0;

                size_t r0 = 0, r1 = 0;
                #pragma omp critical(__pr_block_rss)
                {
                    r0 = rss_bytes();
                    extract_block_slices(
                        i0, i1, BLOCK,
                        deg, bitCount, huffCount,
                        hi, lo,
                        blockHi, blockLo,
                        fallbackBitsOpp,
                        t_deg_blk, t_bit_blk, t_huff_blk,
                        t_hiSlice, t_loSlice,
                        t_baseHiOffset, t_baseLoOffset
                    );
                    r1 = rss_bytes();
                }
                size_t my_block_rss = (r1 > r0 ? (r1 - r0) : 0);
                #pragma omp critical(__pr_peak_block_rss)
                peak_block_rss_bytes = std::max(peak_block_rss_bytes, my_block_rss);

                // Build intra-block prefix offsets once
                const int L = i1 - i0;
                t_offHiBits.assign(L, 0);
                t_offLoBits.assign(L, 0);
                uint64_t accH = 0, accL = 0;
                for (int k = 0; k < L; ++k) {
                    t_offHiBits[k] = accH;
                    t_offLoBits[k] = accL;
                    accH += (uint64_t)t_bit_blk[k];
                    accL += (uint64_t)(t_deg_blk[k] - t_huff_blk[k]) * (uint64_t)fallbackBitsOpp;
                }

                // Process all vertices in this block
                for (int u = i0; u < i1; ++u) {
                    const int du = deg[u];
                    if (du == 0) { local_dangling += pr[u]; continue; }

                    const int k = u - i0;
                    const uint32_t hiOff = t_baseHiOffset + (uint32_t)t_offHiBits[k];
                    const uint32_t loOff = t_baseLoOffset + (uint32_t)t_offLoBits[k];

                    // Decode u from cached slices (no re-extraction)
                    nbrs = decode_single_entry_from_slices(
                        &t_deg_blk[k], &t_bit_blk[k], &t_huff_blk[k],
                        t_hiSlice.data(), t_loSlice.data(),
                        hiOff, loOff,
                        treeOpp, fallbackBitsOpp
                    );

                    const double share = pr[u] / du;
                    // Atomic accumulate to next[v]; stripe-friendly in practice
                    for (int v : nbrs) {
                        if ((unsigned)v < (unsigned)n) {
                            #pragma omp atomic
                            next[v] += share;
                        }
                    }
                }
                // fold thread-local dangling into reduction
                dangling += local_dangling;
                local_dangling = 0.0;
            } // blocks
        } // parallel

        // Damping + dangling distribution
        const double base = (1.0 - alpha) / n;
        const double dang = dangling / n;

        #pragma omp parallel for schedule(static)
        for (int v = 0; v < n; ++v) {
            pr[v] = base + alpha * (next[v] + dang);
        }
    }

    auto t1 = clock::now();
    std::cout << "PageRank (decode, block-parallel cached) finished in "
              << std::chrono::duration<double>(t1 - t0).count() << " s"
              << " | peak block RSS \xCE\x94 = " << peak_block_rss_bytes << " bytes"
              << " | final RSS = " << rss_bytes() << " bytes\n";
}
