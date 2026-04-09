#ifndef CORE_DECOMPOSITION_H
#define CORE_DECOMPOSITION_H

#include <vector>
#include <cstdint>
#include "huffman_tree.h"
using namespace std;
/**
 * @file core_decomposition.h
 * @brief Header file for k-core decomposition on bipartite hypergraphs.
 *
 * The k-core of a graph is the maximal subgraph in which all vertices
 * have degree at least k. In the case of a hypergraph (represented as
 * bipartite), this function computes the vertices that belong to the
 * k-core by iteratively removing vertices and updating incident
 * hyperedges.
 *
 * The result is used for downstream algorithms such as label propagation.
 */

/**
 * @brief Compute the k-core of a hypergraph.
 *
 * This function identifies all vertices that belong to the k-core,
 * defined as the subgraph where every vertex has at least `k_thresh`
 * incident hyperedges.
 *
 * @param v2he     Vertex-to-hyperedge adjacency list.
 * @param he2v     Hyperedge-to-vertex adjacency list.
 * @param k_thresh Minimum degree threshold (k).
 * @return         A binary vector (`char` type), where 1 means in-core and 0 means pruned.
 */
vector<char> computeKCore_onDemand_decodeRandom(
    int n,
    const int* deg,
    const uint16_t* huffCount,
    const uint16_t* bitCount,
    const uint8_t* neighHi,
    const uint8_t* neighLo,
    HuffmanNode* treeOpp,
    int fallbackBitsOpp,
    const uint64_t* blockHi,
    const uint64_t* blockLo,
    int BLOCK,
    int k_thresh
);

vector<char> computeKCore_raw(
    int n,
    const int* deg,
    const int* edges,
    int k_thresh
);
#endif // CORE_DECOMPOSITION_H
