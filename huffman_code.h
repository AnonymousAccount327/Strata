// ===== File: huffman_code.h =====
//
// Description:
// Header file for generating Huffman codes from a binary Huffman tree.
// The core function `genCodes` traverses the Huffman tree and assigns
// a unique binary string (code) to each leaf value, storing them in
// a map for use during encoding.

#ifndef HUFFMAN_CODE_H
#define HUFFMAN_CODE_H

#include <unordered_map>
#include <string>
#include "huffman_tree.h"

/**
 * @brief Generates Huffman codes for all values in the tree.
 *
 * Given a Huffman tree rooted at `root`, this function traverses
 * the tree and fills the map `C` with binary codes for each leaf node.
 *
 * @param root Pointer to the root node of the Huffman tree.
 * @param C    Output map: value → Huffman-encoded binary string.
 */
void genCodes(HuffmanNode* root, std::unordered_map<int, std::string> &C);

#endif  // HUFFMAN_CODE_H
