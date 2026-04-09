// ===== File: hypergraph.cpp =====
//
// Description:
// This file defines the `Hypergraph` class and a function to load a hypergraph
// from a file in the "AdjacencyHypergraph" format. It includes:
//   - Constructor for allocating memory
//   - Destructor for cleanup
//   - Function to read and initialize the hypergraph from disk

#include "hypergraph.h"

// Constructor: initializes the hypergraph structure with given sizes
Hypergraph::Hypergraph(int _nv, int _mv, int _nh, int _mh)
    : nv(_nv), mv(_mv), nh(_nh), mh(_mh)
{
    // Allocate memory for vertex and hyperedge arrays
    edgesV = new int[mv];          // Adjacency list entries for vertices
    edgesH = new int[mh];          // Adjacency list entries for hyperedges
    degreeV = new int[nv]();       // Degree of each vertex
    degreeH = new int[nh]();       // Degree of each hyperedge
}

// Destructor: frees allocated memory
Hypergraph::~Hypergraph() {
    delete[] edgesV;
    delete[] edgesH;
    delete[] degreeV;
    delete[] degreeH;
}

// Reads a hypergraph in the AdjacencyHypergraph format from a file
Hypergraph* readHypergraph(const char *fname) {
    std::ifstream in(fname);
    std::string tag;

    // Validate file format
    in >> tag;
    if (tag != "AdjacencyHypergraph") {
        std::cerr << "Bad file\n";
        exit(1);
    }

    // Read hypergraph dimensions
    int nv, mv, nh, mh;
    in >> nv >> mv >> nh >> mh;

    // Allocate memory for the hypergraph
    auto G = new Hypergraph(nv, mv, nh, mh);

    // Read offsets and adjacency lists
    std::vector<int> offV(nv), offH(nh);
    for (int i = 0; i < nv; i++) in >> offV[i];
    for (int i = 0; i < mv; i++) in >> G->edgesV[i];
    for (int i = 0; i < nh; i++) in >> offH[i];
    for (int i = 0; i < mh; i++) in >> G->edgesH[i];

    // Compute vertex degrees from offsets
    for (int i = 0; i < nv; i++) {
        int next = (i + 1 < nv ? offV[i + 1] : mv);
        G->degreeV[i] = next - offV[i];
    }

    // Compute hyperedge degrees from offsets
    for (int i = 0; i < nh; i++) {
        int next = (i + 1 < nh ? offH[i + 1] : mh);
        G->degreeH[i] = next - offH[i];
    }

    return G;
}
