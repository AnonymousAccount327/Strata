// ============================================================================
// File: decode_blocked.cpp  (self-contained helpers + paged decoder)
// - extract_block_slices: copy only the needed subarrays/bit windows for a block
// - decode_block_partial: decode [i0, i1) using the sliced arrays/bit windows
// - decodeSide (paged): iterate blocks, extract slices, decode, (optional) verify
// ============================================================================

#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>

#include "decode.h"       // readBits(...) declaration + HuffmanNode
#include "huffman_tree.h" // struct HuffmanNode { HuffmanNode* l,*r; int v; ... }

using std::vector;
using std::uint8_t;
using std::uint16_t;
using std::uint64_t;

/**
 * Extract partial arrays and bit slices for a block [i0, i1).
 * Produces:
 *  - deg_blk, bit_blk, huff_blk   (slices of deg/bitCount/huffCount)
 *  - hiSlice, loSlice             (byte windows of neighHi/neighLo covering the block)
 *
 * Assumptions:
 *  - blockHi[k], blockLo[k] store starting bit offsets at entry index k*BLOCK
 *  - Local decoding will start with bit pointers at 0 into these slices
 */
// Extract partial arrays + properly aligned bit slices for [i0,i1)

// Helper function to read `n` bits from array A starting at bit position `pos` (big-endian)
int readBits(const uint8_t *A, long &pos, int n) {
    int x = 0;
    for (int i = 0; i < n; i++) {
        // Extract one bit from byte array A at bit offset pos
        x = (x << 1) | ((A[pos / 8] >> (pos % 8)) & 1);
        pos++;
    }
    return x;
}

size_t bytes_for_bits(uint64_t bits) {
    return static_cast<size_t>((bits + 7ULL) >> 3);
}

void extract_block_slices(
    int i0, int i1,
    int BLOCK,
    const int*       deg,
    const uint16_t*  bitCount,
    const uint16_t*  huffCount,
    const uint8_t*   neighHi,
    const uint8_t*   neighLo,
    const uint64_t*  blockHi,       // bit offsets at block starts
    const uint64_t*  blockLo,       // bit offsets at block starts
    int              fallbackBitsOpp,
    // outputs
    std::vector<int>&       deg_blk,
    std::vector<uint16_t>&  bit_blk,
    std::vector<uint16_t>&  huff_blk,
    std::vector<uint8_t>&   hiSlice,
    std::vector<uint8_t>&   loSlice,
    uint32_t&               hiBitOffset,  // starting bit inside first byte of hiSlice
    uint32_t&               loBitOffset   // starting bit inside first byte of loSlice
) {
    const int len = std::max(0, i1 - i0);
    deg_blk.resize(len);
    bit_blk.resize(len);
    huff_blk.resize(len);
    for (int k = 0; k < len; ++k) {
        deg_blk[k]  = deg[i0 + k];
        bit_blk[k]  = bitCount[i0 + k];
        huff_blk[k] = huffCount[i0 + k];
    }

    // Which block contains i0?
    const int blk = (BLOCK > 0) ? (i0 / BLOCK) : 0;
    const int blkStartIdx = blk * BLOCK;

    // Sum bits for entries inside the block *before i0*
    uint64_t hiBitsBefore = 0, loBitsBefore = 0;
    for (int t = blkStartIdx; t < i0; ++t) {
        hiBitsBefore += bitCount[t];
        loBitsBefore += (uint64_t)(deg[t] - huffCount[t]) * (uint64_t)fallbackBitsOpp;
    }

    // Exact bit starts for entry i0
    const uint64_t hiStartBit = blockHi[blk] + hiBitsBefore;
    const uint64_t loStartBit = blockLo[blk] + loBitsBefore;

    // Total bits needed to cover [i0, i1)
    uint64_t hiBitsThis = 0, loBitsThis = 0;
    for (int i = i0; i < i1; ++i) {
        hiBitsThis += bitCount[i];
        loBitsThis += (uint64_t)(deg[i] - huffCount[i]) * (uint64_t)fallbackBitsOpp;
    }

    // Byte-aligned copy ranges + intra-byte offsets
    const size_t hiSrcByte0 = (size_t)(hiStartBit >> 3);
    const size_t loSrcByte0 = (size_t)(loStartBit >> 3);
    hiBitOffset = (uint32_t)(hiStartBit & 7ULL);
    loBitOffset = (uint32_t)(loStartBit & 7ULL);

    const size_t hiBytes = bytes_for_bits(hiBitOffset + hiBitsThis);
    const size_t loBytes = bytes_for_bits(loBitOffset + loBitsThis);

    hiSlice.resize(hiBytes);
    loSlice.resize(loBytes);

    if (hiBytes) std::memcpy(hiSlice.data(), neighHi + hiSrcByte0, hiBytes);
    if (loBytes) std::memcpy(loSlice.data(), neighLo + loSrcByte0, loBytes);
}


