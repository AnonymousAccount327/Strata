// ==========================================================
// File: BFS_block_memprofile.cpp (RSS-only profiling)
// - Encoded BFS (block-indexed, on-demand decode) with OS-level
//   Resident Set Size (RSS) measurements in BYTES.
//   We report: peak block RSS Δ observed while loading a block.
// - Raw BFS prints RSS at entry/exit and Δ.
// - No allocator/capacity math; no external profilers.
// ==========================================================

#include "BFS.h"
#include <iostream>
#include <queue>
#include <chrono>
#include <vector>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <atomic>
#include <unistd.h>       // sysconf
#include <sys/resource.h>
#include "decode.h"          // extract_block_slices(...), readBits(...)
#include "huffman_tree.h"    // struct HuffmanNode { HuffmanNode *l,*r; int v; }

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

// 1) Use rollup for r0/r1
static size_t rss_bytes_rollup() {
    FILE* f = fopen("/proc/self/smaps_rollup", "r");
    if (!f) return rss_bytes();
    char key[64]; size_t kb = 0;
    while (fscanf(f, "%63s", key) == 1) {
        if (strcmp(key, "Rss:") == 0) { fscanf(f, "%zu", &kb); break; }
        int c; while ((c=fgetc(f))!='\n' && c!=EOF) {}
    }
    fclose(f);
    return kb * 1024ULL;
}

inline size_t round_up(size_t x, size_t u) { return (x + u - 1) / u * u; }

inline void touch_per_page(uint8_t* p, size_t n) {
    if (!p || !n) return;
    long ps = sysconf(_SC_PAGESIZE); if (ps <= 0) ps = 4096;
    for (size_t i = 0; i < n; i += (size_t)ps) { volatile uint8_t s = p[i]; (void)s; }
    volatile uint8_t s = p[n-1]; (void)s;
}


// adapter to reuse your readBits(long&) with a uint64_t bit pointer
static inline int readBits64_local(const uint8_t* buf, uint64_t& bitPtr, int width) {
    long tmp = static_cast<long>(bitPtr);
    int  val = readBits(buf, tmp, width);  // advances tmp
    bitPtr = static_cast<uint64_t>(tmp);
    return val;
}

// ------------------------------------------------------------
// Decode ONE entry from already-extracted, *aligned* slices.
// Inputs:
//   deg_blk[0], bit_blk[0], huff_blk[0] describe this entry
//   hiSlice/loSlice contain only the needed bytes starting at the exact bit
//   hiBitOffset/loBitOffset are the intra-byte starting bit offsets
// Returns: decoded neighbor list for the entry.
// ------------------------------------------------------------
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
        int x = readBits64_local(loSlice, ptrLo_bits, fallbackBitsOpp);
        out.push_back(x);
    }

    return out;
}

// =============================================================
// Single-source BFS (encoded graph, block-indexed, parallel),
// with RSS-only accounting of block loads.
// =============================================================
void runBFSFromSingleSource(
    int n, const int* deg, const int* edges,
    const uint16_t* huffCount, const uint16_t* bitCount,
    const uint8_t* hi, const uint8_t* lo,
    HuffmanNode* treeOpp, int fallbackBitsOpp,
    int src,
    const uint64_t* blockHi, const uint64_t* blockLo, int BLOCK
) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    if (n <= 0 || src < 0 || src >= n) { std::cout << "Single-source BFS: invalid n/src"; return; }
    if (!deg || !huffCount || !bitCount || !hi || !lo || !treeOpp ||
        !blockHi || !blockLo || BLOCK <= 0) {
        std::cerr << "Single-source BFS: missing/invalid inputs";
        return;
    }

    const int INF = INT_MAX/4;
    std::vector<int> dist(n, INF);
    dist[src] = 0;

    // Level-synchronous frontier
    std::vector<int> frontier, nextFrontier;
    frontier.push_back(src);
    int maxLevel = 0, reached = 1;

    // RSS-based accounting for encoded BFS (OS-level resident bytes)
    size_t peak_block_rss_bytes = 0; // max RSS delta seen on a single block load

    while (!frontier.empty()) {
        const int nextLevel = maxLevel + 1;
        nextFrontier.clear();

        // Parallel over the current frontier
        #pragma omp parallel
        {
            // Thread-local block cache (avoid contention)
            int t_curBlk = -1, t_blk_i0 = 0, t_blk_i1 = 0;
            std::vector<int>       t_deg_blk;
            std::vector<uint16_t>  t_bit_blk, t_huff_blk;
            std::vector<uint8_t>   t_hiSlice, t_loSlice;
            uint32_t               t_baseHiOffset = 0, t_baseLoOffset = 0;
            std::vector<uint64_t>  t_offHiBits, t_offLoBits; // helper arrays (not measured specifically)

            std::vector<int> t_localNext;  t_localNext.reserve(1024); // thread-local buffer

            auto ensure_block_loaded = [&](int blk) {
                if (blk == t_curBlk) return;
                t_curBlk = blk;
                t_blk_i0 = blk * BLOCK;
                t_blk_i1 = std::min(n, t_blk_i0 + BLOCK);

                // Release previous block buffers to reduce allocator noise
                t_deg_blk.clear(); t_bit_blk.clear(); t_huff_blk.clear();
                t_hiSlice.clear();  t_loSlice.clear();
                t_deg_blk.shrink_to_fit(); t_bit_blk.shrink_to_fit(); t_huff_blk.shrink_to_fit();
                t_hiSlice.shrink_to_fit();  t_loSlice.shrink_to_fit();

                t_baseHiOffset = t_baseLoOffset = 0;

                // Measure RSS delta around the block load.
                size_t r0 = 0, r1 = 0;
                #pragma omp critical(__bfs_block_rss)
                {
                    r0 = rss_bytes_rollup();
                    extract_block_slices(
                        t_blk_i0, t_blk_i1,
                        BLOCK,
                        deg, bitCount, huffCount,
                        hi, lo,
                        blockHi, blockLo,
                        fallbackBitsOpp,
                        t_deg_blk, t_bit_blk, t_huff_blk,
                        t_hiSlice, t_loSlice,
                        t_baseHiOffset, t_baseLoOffset
                    );
                    r1 = rss_bytes_rollup();
                }
                //cout << "r0: " << r0 << " r1: " << r1 << endl;
                size_t my_block_rss = (r1 > r0 ? (r1 - r0) : 0);
                #pragma omp critical(__bfs_peak_block_rss)
                peak_block_rss_bytes = std::max(peak_block_rss_bytes, my_block_rss);

                // Precompute per-entry bit offsets
                const int L = t_blk_i1 - t_blk_i0;
                t_offHiBits.assign(L, 0);
                t_offLoBits.assign(L, 0);
                uint64_t accH = 0, accL = 0;
                for (int k = 0; k < L; ++k) {
                    t_offHiBits[k] = accH;
                    t_offLoBits[k] = accL;
                    accH += (uint64_t)t_bit_blk[k];
                    accL += (uint64_t)(t_deg_blk[k] - t_huff_blk[k]) * (uint64_t)fallbackBitsOpp;
                }
            };

            #pragma omp for schedule(dynamic, 1024)
            for (int i = 0; i < (int)frontier.size(); ++i) {
                int u = frontier[i];
                if (deg[u] == 0) continue; // isolated

                const int blk = u / BLOCK;
                ensure_block_loaded(blk);

                const int k = u - t_blk_i0;
                const uint32_t hiOff = t_baseHiOffset + (uint32_t)t_offHiBits[k];
                const uint32_t loOff = t_baseLoOffset + (uint32_t)t_offLoBits[k];

                // Decode neighbors of u from the thread-local cached slices
                std::vector<int> nbrs = decode_single_entry_from_slices(
                    &t_deg_blk[k], &t_bit_blk[k], &t_huff_blk[k],
                    t_hiSlice.data(), t_loSlice.data(),
                    hiOff, loOff,
                    treeOpp, fallbackBitsOpp
                );

                // Visit neighbors
                for (int v : nbrs) {
                    if ((unsigned)v >= (unsigned)n) continue;
                    if (dist[v] != INF) continue;
                    // mark visited once
                    #pragma omp critical(__bfs_visit_dist)
                    {
                        if (dist[v] == INF) {
                            dist[v] = nextLevel;
                            t_localNext.push_back(v);
                        }
                    }
                }
            } // for frontier

            // Merge thread-local next frontier
            #pragma omp critical(__bfs_merge_next)
            {
                reached += (int)t_localNext.size();
                nextFrontier.insert(nextFrontier.end(),
                                    t_localNext.begin(), t_localNext.end());
            }
        } // parallel

        // Deduplicate nextFrontier
        std::sort(nextFrontier.begin(), nextFrontier.end());
        nextFrontier.erase(std::unique(nextFrontier.begin(), nextFrontier.end()), nextFrontier.end());

        frontier.swap(nextFrontier);
        if (!frontier.empty()) maxLevel = nextLevel;
    }

    auto t1 = clock::now();

    // Report RSS-based block memory usage (bytes)
    std::cout << std::fixed;
    std::cout << "BFS(encoded): peak block RSS Î = " << peak_block_rss_bytes << " bytes"<<endl
              << " | final RSS = " << rss_bytes() << " bytes"<<endl
              << " | time = " << std::chrono::duration<double>(t1 - t0).count() << " s";
}

// =============================================================
// BFS without decoding: uses deg[] + edges[] directly.
// Prints RSS at entry/exit and delta in BYTES.
// =============================================================
std::vector<int> bfs_single_source_raw(
    int n,
    const int* deg,
    const int* edges,
    int src
) {
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();

    std::vector<int> dist(n, INT_MAX/4);
    if (n <= 0 || src < 0 || src >= n) {
        return dist;
    }

    // Build CSR offsets once (prefix sum)
    std::vector<long long> off(n + 1, 0);
    for (int i = 0; i < n; ++i) off[i + 1] = off[i] + (long long)deg[i];

    size_t r_before = rss_bytes();
    std::cout << "BFS(raw): RSS at entry = " << r_before << " bytes";

    std::queue<int> q;
    dist[src] = 0;
    q.push(src);

    int reached = 1, maxLevel = 0;

    while (!q.empty()) {
        int u = q.front(); q.pop();
        int nextLevel = dist[u] + 1;

        const long long b = off[u], e = off[u + 1];
        for (long long p = b; p < e; ++p) {
            int v = edges[p];
            if ((unsigned)v >= (unsigned)n) continue;
            if (dist[v] != INT_MAX/4) continue;
            dist[v] = nextLevel;
            q.push(v);
            ++reached;
        }
        if (e > b && nextLevel > maxLevel) maxLevel = nextLevel;
    }

    auto t1 = clock::now();
    size_t r_after = rss_bytes();
    std::cout << "BFS(raw): reached=" << reached
              << " | maxLevel=" << maxLevel
              << " | RSS exit = " << r_after << " bytes"
              << " | RSS Î = " << (r_after > r_before ? (r_after - r_before) : 0) << " bytes"
              << " | time=" << std::chrono::duration<double>(t1 - t0).count() << " s";
    return dist;
}
