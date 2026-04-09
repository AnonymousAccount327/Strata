// ===== File: hypergraph.h =====
//
// Description:
// This header defines the Hypergraph structure used for storing hypergraphs 
// in AdjacencyHypergraph format. It includes:
//   - Pointers to adjacency arrays for vertices and hyperedges
//   - Degree arrays for both sides
//   - Constructor and destructor
//   - A loader function to populate a hypergraph from a file

#ifndef HYPERGRAPH_H
#define HYPERGRAPH_H

#include <vector>
#include <fstream>
#include <iostream>

// Hypergraph structure representing a bipartite graph between vertices and hyperedges
struct Hypergraph {
    int nv, mv;   // number of vertices, total vertex edges
    int nh, mh;   // number of hyperedges, total hyperedge edges

    int *edgesV;     // edgesV[i] = vertex-adjacent hyperedge index list
    int *edgesH;     // edgesH[i] = hyperedge-adjacent vertex index list
    int *degreeV;    // degreeV[i] = number of neighbors of vertex i
    int *degreeH;    // degreeH[i] = number of neighbors of hyperedge i

    // Constructor: initializes with given sizes and allocates memory
    Hypergraph(int _nv, int _mv, int _nh, int _mh);

    // Destructor: releases all allocated memory
    ~Hypergraph();
};

// Reads a hypergraph from file in "AdjacencyHypergraph" format
Hypergraph* readHypergraph(const char *fname);

#endif
