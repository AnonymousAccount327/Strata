// ===== File: huffman_code.cpp =====
//
// Description:
// This file implements a helper function `genCodes` that generates the
// Huffman encoding table (map from values to bitstrings) by traversing
// a binary Huffman tree.
//
// The Huffman tree is made up of `HuffmanNode` structs where:
//   - Internal nodes have `v == -1` and non-null `l` (left) and/or `r` (right)
//   - Leaf nodes have `v >= 0` representing the value being encoded
//
// This function performs an iterative DFS traversal to assign a binary string
// to each leaf node (value), storing the result in the output map `C`.
//
// Input:
//   - `root`: Pointer to the root of the Huffman tree
// Output:
//   - `C`: Unordered map (by reference) that will contain the Huffman codes
//          for each value in the tree (e.g., {5: "10", 3: "11", ...})

#include "huffman_code.h"
#include <vector>
#include <utility>   // for std::pair
#include <string>
#include <unordered_map>

/**
 * @brief Recursively generates Huffman codes for all values in a tree.
 *
 * @param root Pointer to the root of the Huffman tree.
 * @param C    Output map: value -> corresponding binary Huffman code as string.
 */
void genCodes(HuffmanNode* root, std::unordered_map<int, std::string> &C) {
    if (!root) return;

    // Stack holds pairs of (node, accumulated code string)
    std::vector<std::pair<HuffmanNode*, std::string>> stk;
    stk.emplace_back(root, "");  // Start with root and empty path

    // Iterative DFS traversal
    while (!stk.empty()) {
        auto [n, s] = stk.back(); stk.pop_back();

        // If node is a leaf, assign its code
        if (n->v >= 0) {
            C[n->v] = s;
        } else {
            // Internal node: push children with updated path
            if (n->r) stk.emplace_back(n->r, s + '1');  // Right child = add '1'
            if (n->l) stk.emplace_back(n->l, s + '0');  // Left child = add '0'
        }
    }
}
