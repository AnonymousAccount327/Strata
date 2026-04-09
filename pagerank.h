#ifndef PAGERANK_H
#define PAGERANK_H

#include <vector>
#include <cstdint>
#include "huffman_tree.h"
/**
 * @file pagerank.h
 * @brief Header file for hypergraph PageRank implementation.
 *
 * This module defines the interface for running the PageRank algorithm
 * on a bipartite hypergraph. The hypergraph is represented as a pair of
 * adjacency lists:
 *   - v2he: vertex-to-hyperedge connectivity
 *   - he2v: hyperedge-to-vertex connectivity
 *
 * PageRank is iteratively computed over both sides, with score updates
 * alternating between vertices and hyperedges.
 *
 */

void runPageRankRaw(
    int n,                    // number of vertices
    const int* deg,           // out-degree per vertex
    const int* edges,         // flat adjacency list, length = sum(deg)
    int iters,
    double alpha /*=0.85*/
);

void runPageRankOnDemand_decodeRandom(
    int n, const int* deg, const int* edges,
    const uint16_t* huffCount, const uint16_t* bitCount,
    const uint8_t* hi, const uint8_t* lo,
    HuffmanNode* treeOpp, int fallbackBitsOpp,
    int iters, double alpha,
    // block index (precomputed in encode; no decoding)
    const uint64_t* blockHi, const uint64_t* blockLo, int BLOCK
);

#endif // PAGERANK_H
