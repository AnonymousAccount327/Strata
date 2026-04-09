// ===== File: huffman_tree.cpp =====
//
// Description:
// This file implements Huffman tree construction and memory management functions.
//
// The Huffman tree is used for compressing data by encoding frequent values
// with shorter bit strings. This file includes:
//   - A custom HuffmanNode constructor
//   - The `buildTree` function that selects the top-k most frequent symbols
//     for Huffman encoding and computes the number of fallback bits needed
//     to encode the rest
//   - Utility functions to measure memory usage (`treeMem`) and deallocate
//     the tree (`freeTree`)

#include "huffman_tree.h"
#include <queue>
#include <algorithm>

// Comparator for max-heap: used to select top-k frequent symbols
struct CmpMax {
    bool operator()(HuffmanNode* a, HuffmanNode* b) const {
        return a->f < b->f; // higher freq = higher priority
    }
};

// Comparator for min-heap: used to build Huffman tree
struct CmpMin {
    bool operator()(HuffmanNode* a, HuffmanNode* b) const {
        return a->f > b->f; // lower freq = higher priority
    }
};

// Constructor for HuffmanNode
HuffmanNode::HuffmanNode(int _v, int _f)
    : v(_v), f(_f), l(nullptr), r(nullptr) {}

// Builds a Huffman tree from frequency array `freq` of length `n`
// - Only encodes the top `pct` fraction of most frequent items
// - Returns the root of the tree
// - Computes fallbackBits: the number of bits needed to encode the remaining symbols
HuffmanNode* buildTree(const int* freq, int n, double pct, int& fallbackBits) {
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CmpMax> maxQ;

    // Step 1: Push all symbols into max-heap based on frequency
    for (int i = 0; i < n; ++i) {
        maxQ.push(new HuffmanNode(i, freq[i]));
    }

    // Step 2: Extract top-k frequent items for Huffman encoding
    int k = static_cast<int>(n * pct + 0.5);
    std::priority_queue<HuffmanNode*, std::vector<HuffmanNode*>, CmpMin> minQ;
    for (int i = 0; i < k && !maxQ.empty(); ++i) {
        minQ.push(maxQ.top());
        maxQ.pop();
    }

    // Step 3: Determine how many fallback bits are needed for non-Huffman items
    int maxSym = 0;
    while (!maxQ.empty()) {
        maxSym = std::max(maxSym, maxQ.top()->v);
        delete maxQ.top(); // Clean up unused nodes
        maxQ.pop();
    }

    fallbackBits = 0;
    while ((1 << fallbackBits) <= maxSym) {
        ++fallbackBits;
    }
    if (fallbackBits == 0) fallbackBits = 1;

    // Step 4: Build the Huffman tree using the min-heap
    while (minQ.size() > 1) {
        HuffmanNode* a = minQ.top(); minQ.pop();
        HuffmanNode* b = minQ.top(); minQ.pop();
        HuffmanNode* m = new HuffmanNode(-1, a->f + b->f); // Internal node
        m->l = a;
        m->r = b;
        minQ.push(m);
    }

    return minQ.empty() ? nullptr : minQ.top();
}

// Recursively compute memory usage of the tree
long treeMem(HuffmanNode* n) {
    if (!n) return 0;
    return sizeof(*n) + treeMem(n->l) + treeMem(n->r);
}

// Recursively deallocate the Huffman tree
void freeTree(HuffmanNode* n) {
    if (!n) return;
    freeTree(n->l);
    freeTree(n->r);
    delete n;
}
