#include "computeKCore.h"
#include <queue>
#include <iostream>
#include <chrono>
#include <sys/resource.h>
#include <unistd.h>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <cstdint>
#include "decode.h"
#include "huffman_tree.h"

using namespace std;

// ------------------------ RSS helpers ------------------------
static inline long rss_kb() {
    rusage ru;
    getrusage(RUSAGE_SELF, &ru);
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
    if (fscanf(f, "%lu %lu", &dummy, &pages) != 2) {
        fclose(f);
        return (size_t)rss_kb() * 1024ull;
    }
    fclose(f);
    long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0) ps = 4096;
    return (size_t)pages * (size_t)ps;
#else
    return (size_t)rss_kb() * 1024ull;
#endif
}

// --- adapter to use your readBits(long&) with a uint64_t pointer ---
static inline int readBits64_local(const uint8_t* buf, uint64_t& bitPtr, int width) {
    long tmp = static_cast<long>(bitPtr);
    int  val = readBits(buf, tmp, width);  // advances tmp
    bitPtr = static_cast<uint64_t>(tmp);
    return val;
}

// Decode ONE entry from already-extracted, aligned slices.
static std::vector<int> decode_single_entry_from_slices(
    const int*       deg_blk,
    const uint16_t*  bit_blk,
    const uint16_t*  huff_blk,
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
        int x = readBits64_local(loSlice, ptrLo_bits, fallbackBitsOpp);
        out.push_back(x);
    }

    return out;
}

// --- K-core on the encoded graph (decodes neighbors on demand via decodeRandom) ---
std::vector<char> computeKCore_onDemand_decodeRandom(
    int n,
    const int* deg,
    const uint16_t* huffCount,
    const uint16_t* bitCount,
    const uint8_t* neighHi,
    const uint8_t* neighLo,
    HuffmanNode* treeOpp,
    int fallbackBitsOpp,
    // block index built at encode-time
    const uint64_t* blockHi,
    const uint64_t* blockLo,
    int BLOCK,
    int k_thresh
) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    if (n <= 0) return {};

    // ---- working degree (mutable), removal flags, initial frontier ----
    std::vector<int>  curDeg(n);
    std::vector<char> removed(n, 0);
    std::vector<int>  R; R.reserve(1024);

    for (int v = 0; v < n; ++v) {
        curDeg[v] = deg[v];
        if (curDeg[v] < k_thresh) { removed[v] = 1; R.push_back(v); }
    }

    // RSS-based accounting for per-vertex slice extraction
    size_t peak_slice_rss_bytes = 0; // max Δ per extract_block_slices([v,v+1))

    // ---- peeling rounds ----
    while (!R.empty()) {
        std::vector<int> nextR;

        #pragma omp parallel
        {
            // Thread-local buffer for next round
            std::vector<int> localNext;
            localNext.reserve(256);

            // Each thread extracts only the needed block slices for one vertex
            #pragma omp for schedule(dynamic, 1024)
            for (int idx = 0; idx < (int)R.size(); ++idx) {
                const int v = R[idx];
                if (deg[v] == 0) continue;  // isolated

                // --- per-vertex block slice extraction [v, v+1) ---
                std::vector<int>       deg_blk;
                std::vector<uint16_t>  bit_blk, huff_blk;
                std::vector<uint8_t>   hiSlice, loSlice;
                uint32_t hiBitOffset = 0, loBitOffset = 0;

                // Measure RSS delta around the slice extraction; serialize sampling
                size_t r0 = 0, r1 = 0;
                #pragma omp critical(__kcore_block_rss)
                {
                    r0 = rss_bytes();
                    extract_block_slices(
                        /*i0=*/v, /*i1=*/v+1,
                        BLOCK,
                        deg, bitCount, huffCount,
                        neighHi, neighLo,
                        blockHi, blockLo,
                        fallbackBitsOpp,
                        deg_blk, bit_blk, huff_blk,
                        hiSlice, loSlice,
                        hiBitOffset, loBitOffset
                    );
                    r1 = rss_bytes();
                }
                size_t my_slice_rss = (r1 > r0 ? (r1 - r0) : 0);
                #pragma omp critical(__kcore_peak_slice_rss)
                peak_slice_rss_bytes = std::max(peak_slice_rss_bytes, my_slice_rss);

                // --- decode exactly one entry (v) from the slices ---
                std::vector<int> nbrs = decode_single_entry_from_slices(
                    deg_blk.data(), bit_blk.data(), huff_blk.data(),
                    hiSlice.data(),  loSlice.data(),
                    hiBitOffset,     loBitOffset,
                    treeOpp, fallbackBitsOpp
                );

                // --- update neighbors ---
                for (int u : nbrs) {
                    if ((unsigned)u >= (unsigned)n) continue;
                    if (removed[u]) continue;

                    int old;
                    #pragma omp atomic capture
                    { old = curDeg[u]; curDeg[u] = old - 1; }

                    if (old == k_thresh) {
                        localNext.push_back(u);
                    }
                }
            } // omp for

            #pragma omp critical
            nextR.insert(nextR.end(), localNext.begin(), localNext.end());
        } // omp parallel

        if (nextR.empty()) break;

        // deduplicate and finalize removals
        std::sort(nextR.begin(), nextR.end());
        nextR.erase(std::unique(nextR.begin(), nextR.end()), nextR.end());

        int w = 0;
        for (int u : nextR) {
            if (!removed[u]) {
                removed[u] = 1;
                nextR[w++] = u;
            }
        }
        nextR.resize(w);

        R.swap(nextR);
    }

    // ---- build in-core mask ----
    std::vector<char> inCore(n, 0);
    int remain = 0;
    for (int v = 0; v < n; ++v) { inCore[v] = !removed[v]; remain += inCore[v]; }

    auto t1 = clock::now();
    std::cout << "KCore(encoded): peak slice RSS \xCE\x94 = " << peak_slice_rss_bytes
              << " bytes | final RSS = " << rss_bytes()
              << " bytes | time = " << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    return inCore;
}

// --- K-core on the raw (uncompressed) representation ---
// edges is the flat adjacency list of length sum(deg)
std::vector<char> computeKCore_raw(
    int n,
    const int* deg,
    const int* edges,
    int k_thresh
) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    if (n <= 0) return {};

    size_t r_entry = rss_bytes();

    // prefix offsets to locate neighbors of v in O(1)
    std::vector<uint64_t> off(n + 1, 0);
    for (int v = 0; v < n; ++v) off[v + 1] = off[v] + (uint64_t)deg[v];

    std::vector<int>  curDeg(n);
    std::vector<char> removed(n, 0);
    for (int v = 0; v < n; ++v) curDeg[v] = deg[v];

    std::vector<int> R; R.reserve(1024);
    for (int v = 0; v < n; ++v) {
        if (curDeg[v] < k_thresh) { removed[v] = 1; R.push_back(v); }
    }

    while (!R.empty()) {
        std::vector<int> nextR;

        #pragma omp parallel
        {
            std::vector<int> localNext;
            localNext.reserve(256);

            #pragma omp for schedule(dynamic, 1024)
            for (int i = 0; i < (int)R.size(); ++i) {
                int v = R[i];
                if (deg[v] == 0) continue;

                const uint64_t s = off[v];
                const uint64_t e = off[v + 1];

                for (uint64_t p = s; p < e; ++p) {
                    int u = edges[p];
                    if ((unsigned)u >= (unsigned)n) continue;
                    if (removed[u]) continue;

                    int old;
                    #pragma omp atomic capture
                    { old = curDeg[u]; curDeg[u] = old - 1; }

                    if (old == k_thresh) localNext.push_back(u);
                }
            }

            #pragma omp critical
            nextR.insert(nextR.end(), localNext.begin(), localNext.end());
        } // parallel

        if (nextR.empty()) break;

        std::sort(nextR.begin(), nextR.end());
        nextR.erase(std::unique(nextR.begin(), nextR.end()), nextR.end());

        int w = 0;
        for (int u : nextR) {
            if (!removed[u]) { removed[u] = 1; nextR[w++] = u; }
        }
        nextR.resize(w);

        R.swap(nextR);
    }

    std::vector<char> inCore(n, 0);
    int remain = 0;
    for (int v = 0; v < n; ++v) { inCore[v] = !removed[v]; remain += inCore[v]; }

    auto t1 = clock::now();
    size_t r_exit = rss_bytes();
    std::cout << "KCore(raw): remain=" << remain << " / " << n
              << " | RSS entry = " << r_entry
              << " bytes | RSS exit = " << r_exit
              << " bytes | RSS \xCE\x94 = " << (r_exit > r_entry ? (r_exit - r_entry) : 0)
              << " bytes | time = " << std::chrono::duration<double>(t1 - t0).count()
              << " s\n";

    return inCore;
}
