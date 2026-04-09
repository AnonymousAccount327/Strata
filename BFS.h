#ifndef BFS_H
#define BFS_H

#include <vector>
#include "hypergraph.h"
#include "huffman_tree.h"
using namespace std;
/**
 * @file BFS.h
 * @brief Header for BFS-based graph algorithms on hypergraphs.
 *
 * This module provides utilities to run breadth-first search (BFS),
 * connected component analysis, and full-graph traversals on hypergraphs
 * represented either as flat arrays (CSR-style) or as adjacency lists.
 */

/**
 * @brief Run a BFS traversal starting from a fixed vertex (e.g., vertex 5050).
 *
 * Traverses the hypergraph using alternating vertex-hyperedge hops.
 *
 * @param nv       Number of vertices.
 * @param nh       Number of hyperedges.
 * @param degreeV  Vertex degrees.
 * @param edgesV   Vertex→hyperedge flattened edge list.
 * @param degreeH  Hyperedge degrees.
 * @param edgesH   Hyperedge→vertex flattened edge list.
 * @param offsetV  CSR-style offset array for vertex→hyperedges.
 * @param offsetH  CSR-style offset array for hyperedge→vertices.
 */

 void runBFSFromSingleSource(
    int n, const int* deg, const int* edges,
    const uint16_t* huffCount, const uint16_t* bitCount,
    const uint8_t* hi, const uint8_t* lo,
    HuffmanNode* treeOpp, int fallbackBitsOpp,
    int src,
    const uint64_t* blockHi, const uint64_t* blockLo, int BLOCK   // <- blocks from encoder
);

// ---- BFS (raw / decoded) declarations ----
vector<int> bfs_single_source_raw(
    int n,
    const int* deg,
    const int* edges,
    int src
);

#endif // BFS_H
