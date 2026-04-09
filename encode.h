// ===== File: encode.h =====
//
// Description:
// Header for the hybrid encoding function used in hypergraph compression.
//
// The `encodeSide` function compresses the adjacency information for a set of
// vertices or hyperedges. It uses a two-tier scheme:
//   - Huffman encoding (via a provided code map) for frequent values
//   - Fixed-width binary fallback encoding for uncommon values
//
// It produces two bitstreams (`neighHi`, `neighLo`) and an auxiliary
// array (`bitCount`) that tracks the number of Huffman bits used per element.
//
// Output bitstreams are packed into `uint8_t` arrays.
//
// Dependencies:
//   - Requires C++ STL (unordered_map, string, vector, etc.)

#ifndef ENCODE_H
#define ENCODE_H

#include <unordered_map>
#include <cstdint>
#include <utility>
#include <string>
using namespace std;

/**
 * @brief Encodes the adjacency list using hybrid Huffman and fallback encoding.
 * 
 * @param n               Number of entries (vertices or hyperedges) to encode
 * @param deg             Array of size n, degree of each entry
 * @param edges           Flat array of neighbors to encode (sum of all degrees in size)
 * @param codesOpp        Huffman code table mapping neighbor values to bitstrings
 * @param fallbackBitsOpp Number of bits for fixed-width fallback encoding
 * @param bitCount        (Output) Array of size n, stores Huffman bit count for each entry
 * @param neighHi         (Output) Byte array of packed Huffman-encoded bits
 * @param neighLo         (Output) Byte array of packed fallback-encoded bits
 * 
 * @return std::pair<long, long>
 *         First: Total number of bits in Huffman bitstream (neighHi)
 *         Second: Total number of bits in fallback bitstream (neighLo)
 */

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
    int B,                   
    uint64_t *&blockHi,      
    uint64_t *&blockLo,     
    int *nBlocks              
);

#endif  // ENCODE_H
