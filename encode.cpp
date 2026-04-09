// ===== File: encode.cpp =====
//
// Description:
// This file implements the hybrid encoding routine for hypergraph compression.
// Each neighbor (or edge) is either encoded using:
//   - Huffman coding (via `codesOpp`) for high-frequency values
//   - Fixed-width binary encoding (`fallbackBitsOpp`) for others
//
// Two separate bitstreams are produced:
//   - `neighHi`: Huffman-encoded bits (packed into a byte array)
//   - `neighLo`: Fallback-encoded bits (packed into a byte array)
//
// The function also computes and returns:
//   - `bitCount[i]`: number of bits used by Huffman codes for entry `i`
//   - `(bitsH, bitsL)`: total bits used for Huffman and fallback encoding respectively
//
// Inputs:
//   - n:            Number of vertices (or hyperedges) to encode
//   - deg:          Degree of each entry (array of size n)
//   - edges:        Flat array of neighbor indices
//   - codesOpp:     Huffman code table mapping values to binary strings
//   - fallbackBitsOpp: Number of bits for fixed-width fallback encoding
//
// Outputs (allocated within function):
//   - bitCount:     Array of size n, storing total Huffman bits used per entry
//   - huffCount:    Array of size n, storing number of Huffman-coded symbols per entry
//   - neighHi:      Byte array of packed Huffman-encoded bits
//   - neighLo:      Byte array of packed fallback-encoded bits
//
// Return value:
//   - Pair (bitsH, bitsL): total number of bits in neighHi and neighLo respectively
//

// ===== File: encode.cpp (patched) =====

#include "encode.h"
#include <vector>
#include <unordered_map>
#include <sys/resource.h>
#include <unistd.h>
#include <chrono>
#include <malloc.h>
#include <cstdint>

using namespace std;

pair<long, long> encodeSide(
    int n,
    const int *deg,
    const int *edges,
    const unordered_map<int, std::string> &codesOpp,
    int fallbackBitsOpp,
    uint16_t *&huffCount,
    uint16_t *&bitCount,
    uint8_t *&neighHi,
    uint8_t *&neighLo,
    // --- NEW: block index outputs ---
    int B,                    // block size (e.g., 512 or 1024)
    uint64_t *&blockHi,       // size = nBlocks, bit offset into neighHi at block start
    uint64_t *&blockLo,       // size = nBlocks, bit offset into neighLo at block start
    int *nBlocks              // number of blocks produced
) {
    vector<bool> HF; // Huffman bit stream (as a vector of bits)
    vector<bool> LF; // Fallback bit stream (as a vector of bits)

    bitCount  = new uint16_t[n]();  // per-entry Huffman bit lengths
    huffCount = new uint16_t[n]();  // per-entry # of Huffman-coded symbols

    long ptr   = 0;  // index into flat edges[]
    long bitsH = 0;  // total Huffman bits so far
    long bitsL = 0;  // total fallback bits so far

    // --- NEW: allocate block arrays ---
    if (B <= 0) B = 1024;                  // sane default
    *nBlocks  = (n + B - 1) / B;
    blockHi  = *nBlocks ? new uint64_t[*nBlocks] : nullptr;
    blockLo  = *nBlocks ? new uint64_t[*nBlocks] : nullptr;

    // Loop through each entry
    for (int i = 0; i < n; i++) {
        // --- NEW: record block start offsets ---
        if (B > 0 && (i % B) == 0) {
            int b = i / B;
            // Record CURRENT totals as the starting offsets of this block
            if (blockHi) blockHi[b] = static_cast<uint64_t>(bitsH);
            if (blockLo) blockLo[b] = static_cast<uint64_t>(bitsL);
        }

        int huffNum = 0;

        // Encode neighbors of entry i
        for (int j = 0; j < deg[i]; j++, ptr++) {
            int nei = edges[ptr];

            // Try Huffman encoding first
            auto it = codesOpp.find(nei);
            if (it != codesOpp.end()) {
                // Encode using Huffman (append each bit)
                const std::string &code = it->second;
                for (char c : code) {
                    HF.push_back(c == '1');
                    bitsH++;
                }
                bitCount[i] += static_cast<uint16_t>(code.size());  // track bits used for i
                huffNum++;
            } else {
                // Fallback to fixed-width binary encoding (MSB-first)
                unsigned x = static_cast<unsigned>(nei);
                for (int b = fallbackBitsOpp - 1; b >= 0; --b) {
                    LF.push_back((x >> b) & 1U);
                    bitsL++;
                }
            }
        }
        huffCount[i] = static_cast<uint16_t>(huffNum);
    }

    // Compute number of bytes needed for each stream (round up to byte boundary)
    long bytesH = (bitsH + 7) / 8;
    long bytesL = (bitsL + 7) / 8;

    // Allocate output byte arrays and initialize to 0
    neighHi = bytesH ? new uint8_t[bytesH]() : nullptr;
    neighLo = bytesL ? new uint8_t[bytesL]() : nullptr;

    // Pack bits into bytes for Huffman stream
    for (long i = 0; i < bitsH; i++) {
        if (HF[i]) neighHi[i / 8] |= static_cast<uint8_t>(1U << (i % 8));
    }

    // Pack bits into bytes for fallback stream
    for (long i = 0; i < bitsL; i++) {
        if (LF[i]) neighLo[i / 8] |= static_cast<uint8_t>(1U << (i % 8));
    }

    return {bitsH, bitsL};  // total bits in each stream
}
