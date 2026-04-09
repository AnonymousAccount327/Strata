// ===== File: decode.h =====
//
// Description:
// Interface for the hybrid decoding routine used in compressed hypergraph processing.
// This decoder reconstructs the adjacency list of vertices or hyperedges from a two-tier
// encoded format:
//   - `neighHi`: Encoded using Huffman tree compression for frequent elements.
//   - `neighLo`: Encoded using fixed-width fallback encoding for rare elements.
// It also performs a checksum comparison (sum-check) against the original uncompressed
// values to ensure decoding correctness.
//
// Usage:
// Include this header when calling `decodeSide()` to reconstruct the decoded graph
// for tasks such as BFS or other graph traversals.
//
// Dependencies:
// - Requires the Huffman tree structure defined in "huffman_tree.h"
//

#ifndef DECODE_H
#define DECODE_H

#include <cstdint>
#include <vector>
#include "huffman_tree.h"
using namespace std;
// Function: decodeSide
//
// Parameters:
//   - n:              Number of vertices (or hyperedges) to decode
//   - deg:            Array of size n, degree of each vertex (or size of each hyperedge)
//   - edges:          Flat array of original uncompressed neighbor values (for verification)
//   - bitCount:       Array of size n, total Huffman bits assigned to each entry
//   - neighHi:        Bitstream storing Huffman-encoded values (packed into bytes)
//   - neighLo:        Bitstream storing fallback-encoded values (fixed width per value)
//   - treeOpp:        Root pointer to Huffman tree used for decoding
//   - fallbackBitsOpp:Number of bits per fallback value (typically log2 of range)
//
// Returns:
//   A vector of n vectors, where each inner vector contains the decoded neighbors
//   of the corresponding vertex/hyperedge.
//
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
    uint32_t&               hiBitOffset,  // NEW: starting bit inside first byte of hiSlice
    uint32_t&               loBitOffset   // NEW: starting bit inside first byte of loSlice
);

int readBits(const uint8_t *A, long &pos, int n);

#endif  // DECODE_H
