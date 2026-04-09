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
#include <cstdint>     // uint8_t, uint16_t, uint64_t
#include <cstdio>
#include <cstring>

#include "hypergraph.h"
#include "huffman_tree.h"
#include "huffman_code.h"
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
    return 2 * sizeof(int*)            // deg, bitCount
         + sizeof(uint16_t*)           // huffNum
         + 2 * sizeof(uint8_t*)        // neighHi, neighLo
         + sizeof(int) * n             // deg
         + sizeof(int) * n             // bitCount
         + bH * sizeof(uint8_t)        // neighHi
         + bL * sizeof(uint8_t)        // neighLo
         + sizeof(uint16_t) * n        // huffCount
         + 2 * sizeof(uint64_t) * ((n + 1024 - 1) / 1024);
}

static size_t footprint_one(int n, long bitsH, long bitsL) {
    long bH = (bitsH + 7) / 8, bL = (bitsL + 7) / 8;
    return 2 * sizeof(int*)            // deg, bitCount
         + sizeof(uint16_t*)           // huffNum
         + 2 * sizeof(uint8_t*)        // neighHi, neighLo
         + sizeof(uint16_t) * n        // deg
         + sizeof(int) * n             // bitCount
         + bH * sizeof(uint8_t)        // neighHi
         + bL * sizeof(uint8_t)        // neighLo
         + sizeof(uint16_t) * n        // huffCount
         + 2 * sizeof(uint64_t) * ((n + 1024 - 1) / 1024);
}

static size_t footprint_both(int n, long bitsH, long bitsL) {
    long bH = (bitsH + 7) / 8, bL = (bitsL + 7) / 8;
    return 2 * sizeof(uint16_t*)       // deg, bitCount
         + sizeof(uint16_t*)           // huffNum
         + 2 * sizeof(uint8_t*)        // neighHi, neighLo
         + sizeof(uint16_t) * n        // deg
         + sizeof(uint16_t) * n        // bitCount
         + bH * sizeof(uint8_t)        // neighHi
         + bL * sizeof(uint8_t)        // neighLo
         + sizeof(uint16_t) * n        // huffCount
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

    return (size_t)n * sizeof(int)      // deg[]
         + (size_t)m * sizeof(int);     // edges[]
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
    // Microbenchmark: blocked random-access decode latency
    // -------------------------------------------------------------------------
    {
        using clock = std::chrono::high_resolution_clock;
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

        const int NUM_QUERIES = std::min(n, 1000);
        std::vector<int> queries(NUM_QUERIES);
        for (int i = 0; i < NUM_QUERIES; ++i) {
            queries[i] = (long long)i * n / NUM_QUERIES;
        }

        const int* degSide = encodeH ? G->degreeH : G->degreeV;
        HuffmanNode* treeOpp = encodeH ? tV : tH;
        int fallbackBitsOpp = encodeH ? fbV : fbH;

        long long total_setup_ns = 0;
        long long total_decode_ns = 0;
        long long total_e2e_ns = 0;
        long long max_setup_ns = 0;
        long long max_decode_ns = 0;
        long long max_e2e_ns = 0;

        for (int u : queries) {
            const int blk = u / BLOCK;
            const int i0 = blk * BLOCK;
            const int i1 = std::min(n, i0 + BLOCK);

            std::vector<int>       deg_blk;
            std::vector<uint16_t>  bit_blk, huff_blk;
            std::vector<uint8_t>   hiSlice, loSlice;
            uint32_t               hiBitOffset = 0, loBitOffset = 0;

            auto t0 = clock::now();

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

            auto t1 = clock::now();

            const int k = u - i0;
            const uint32_t hiOff = hiBitOffset + (uint32_t)offHiBits[k];
            const uint32_t loOff = loBitOffset + (uint32_t)offLoBits[k];

            auto t2 = clock::now();

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

            auto t3 = clock::now();
            (void)nbrs;

            long long setup_ns  = std::chrono::duration_cast<ns>(t1 - t0).count();
            long long decode_ns = std::chrono::duration_cast<ns>(t3 - t2).count();
            long long e2e_ns    = std::chrono::duration_cast<ns>(t3 - t0).count();

            total_setup_ns  += setup_ns;
            total_decode_ns += decode_ns;
            total_e2e_ns    += e2e_ns;

            max_setup_ns  = std::max(max_setup_ns, setup_ns);
            max_decode_ns = std::max(max_decode_ns, decode_ns);
            max_e2e_ns    = std::max(max_e2e_ns, e2e_ns);
        }

        cout << "Blocked decode microbenchmark over " << NUM_QUERIES << " queries:\n";
        cout << "  Avg block setup time   = "
             << (double)total_setup_ns / NUM_QUERIES / 1000.0 << " us\n";
        cout << "  Avg local decode time  = "
             << (double)total_decode_ns / NUM_QUERIES / 1000.0 << " us\n";
        cout << "  Avg end-to-end time    = "
             << (double)total_e2e_ns / NUM_QUERIES / 1000.0 << " us\n";
        cout << "  Max block setup time   = "
             << (double)max_setup_ns / 1000.0 << " us\n";
        cout << "  Max local decode time  = "
             << (double)max_decode_ns / 1000.0 << " us\n";
        cout << "  Max end-to-end time    = "
             << (double)max_e2e_ns / 1000.0 << " us\n";
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

        // Fully decoded representation
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

    /*
    auto t_bfs0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        runBFSFromSingleSource(
            encodeH ? nh : nv,
            encodeH ? G->degreeH : G->degreeV,
            encodeH ? G->edgesH  : G->edgesV,
            huffCount, bitCount, hi, lo,
            encodeH ? tV : tH, encodeH ? fbV : fbH,
            i,
            blockHi, blockLo, BLOCK
        );
    }
    auto t_bfs1 = chrono::high_resolution_clock::now();
    cout << "BFS(compressed) Time: "
         << chrono::duration<double>(t_bfs1 - t_bfs0).count()
         << " s\n";

    auto t_bfs2 = chrono::high_resolution_clock::now();
    for (int i = 0; i < 3; i++) {
        auto dist_raw = bfs_single_source_raw(
            encodeH ? nh : nv,
            encodeH ? G->degreeH : G->degreeV,
            encodeH ? G->edgesH  : G->edgesV,
            i
        );
        (void)dist_raw;
    }
    auto t_bfs3 = chrono::high_resolution_clock::now();
    cout << "BFS(raw) Time: "
         << chrono::duration<double>(t_bfs3 - t_bfs2).count()
         << " s\n";

    size_t raw_bytes = compute_raw_side_bytes(
        encodeH, nv, nh, G->degreeV, G->degreeH
    );
    (void)raw_bytes;

    auto t_kcore0 = chrono::high_resolution_clock::now();
    for (int i = 1; i <= 5; i++) {
        auto inCore_dec = computeKCore_onDemand_decodeRandom(
            n,
            encodeH ? G->degreeH : G->degreeV,
            huffCount, bitCount,
            hi, lo,
            encodeH ? tV : tH, encodeH ? fbV : fbH,
            blockHi, blockLo, BLOCK,
            i
        );
        (void)inCore_dec;
    }
    auto t_kcore1 = chrono::high_resolution_clock::now();
    cout << "K-core(compressed) Time: "
         << chrono::duration<double>(t_kcore1 - t_kcore0).count()
         << " s\n";

    auto t_kcore2 = chrono::high_resolution_clock::now();
    for (int i = 1; i <= 2; i++) {
        auto inCore_raw = computeKCore_raw(
            n,
            encodeH ? G->degreeH : G->degreeV,
            encodeH ? G->edgesH : G->edgesV,
            i
        );
        (void)inCore_raw;
    }
    auto t_kcore3 = chrono::high_resolution_clock::now();
    cout << "K-core(raw) Time: "
         << chrono::duration<double>(t_kcore3 - t_kcore2).count()
         << " s\n";

    runPageRankOnDemand_decodeRandom(
        encodeH ? nh : nv,
        encodeH ? G->degreeH : G->degreeV,
        encodeH ? G->edgesH  : G->edgesV,
        huffCount, bitCount,
        hi, lo,
        encodeH ? tV : tH, encodeH ? fbV : fbH,
        pr_iters, 0.85,
        blockHi, blockLo, BLOCK
    );
    */

    runPageRankRaw(
        encodeH ? nh : nv,
        encodeH ? G->degreeH : G->degreeV,
        encodeH ? G->edgesH  : G->edgesV,
        pr_iters,
        0.85
    );

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