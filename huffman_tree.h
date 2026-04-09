// ===== File: huffman_tree.h =====
//
// Description:
// Header file for Huffman tree construction and management.
// Defines the `HuffmanNode` structure and utility functions to:
//   - Build a Huffman tree for the top-k most frequent items
//   - Compute memory usage of the tree
//   - Free the memory used by the tree

#ifndef HUFFMAN_TREE_H
#define HUFFMAN_TREE_H

#include <queue>

// Structure representing a node in the Huffman tree
struct HuffmanNode {
    int v;         // Symbol value (non-negative for leaves, -1 for internal nodes)
    int f;         // Frequency of the symbol or combined frequency
    HuffmanNode *l; // Pointer to left child
    HuffmanNode *r; // Pointer to right child

    // Constructor
    HuffmanNode(int _v, int _f);
};

/**
 * Constructs a Huffman tree from a frequency array.
 * 
 * @param freq          Array of frequencies, one for each symbol [0, n)
 * @param n             Number of symbols
 * @param pct           Percentage (0 < pct ≤ 1.0) of most frequent items to include in the tree
 * @param fallbackBits  Output parameter: number of bits needed to encode remaining (non-Huffman) items
 * @return              Pointer to root of the Huffman tree
 */
HuffmanNode* buildTree(const int *freq, int n, double pct, int &fallbackBits);

/**
 * Recursively computes the memory (in bytes) used by the tree.
 *
 * @param n  Pointer to root of Huffman tree
 * @return   Total memory used by tree nodes (in bytes)
 */
long treeMem(HuffmanNode* n);

/**
 * Recursively deallocates the entire Huffman tree.
 *
 * @param n  Pointer to root of Huffman tree
 */
void freeTree(HuffmanNode* n);

#endif // HUFFMAN_TREE_H
