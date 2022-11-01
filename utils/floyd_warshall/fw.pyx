# Author: Jake Vanderplas  -- <vanderplas@astro.washington.edu>
# License: BSD, (C) 2012

import numpy as np
cimport numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph._validation import validate_graph

cimport cython
from libc cimport stdlib

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# Fused type for int32 and int64
ctypedef fused int32_or_int64:
    np.int32_t
    np.int64_t

# EPS is the precision of DTYPE
DEF DTYPE_EPS = 1E-15

# NULL_IDX is the index used in predecessor matrices to store a non-path
DEF NULL_IDX = -9999

cpdef optimised_floyd_warshall(csgraph):
    csgraph = validate_graph(csgraph, False, dense_output=False)
    cdef int N = csgraph.shape[0]
    cdef np.ndarray distances = np.ones((N,N),dtype=ITYPE)*np.inf
    cdef int node=0
    cdef ssize_t i_start =0

    cdef np.ndarray node_list = np.empty(N, dtype=ITYPE)
    cdef np.ndarray predecessors = np.empty(N, dtype=ITYPE)

    csgraph_T = csgraph.T.tocsr()

    for i_start in range(N):
        distances[i_start,i_start]=0
        node_list.fill(NULL_IDX)
        predecessors.fill(NULL_IDX)
        length = _breadth_first_undirected(i_start,
                                           csgraph.indices, csgraph.indptr,
                                           csgraph_T.indices, csgraph_T.indptr,
                                           node_list, predecessors)
        for n in range(1,length):
            node=node_list[n]
            prev=predecessors[node]
            distances[i_start,node]=distances[i_start,prev]+1

    return distances


cdef unsigned int _breadth_first_undirected(
                           unsigned int head_node,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indices1,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr1,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indices2,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] indptr2,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] node_list,
                           np.ndarray[ITYPE_t, ndim=1, mode='c'] predecessors):
                           
    cdef unsigned int i, pnode, cnode
    cdef unsigned int i_nl, i_nl_end
    cdef unsigned int N = node_list.shape[0]

    node_list[0] = head_node
    i_nl = 0
    i_nl_end = 1

    while i_nl < i_nl_end:
        pnode = node_list[i_nl]

        for i in range(indptr1[pnode], indptr1[pnode + 1]):
            cnode = indices1[i]
            if (cnode == head_node):
                continue
            elif (predecessors[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                predecessors[cnode] = pnode
                i_nl_end += 1

        for i in range(indptr2[pnode], indptr2[pnode + 1]):
            cnode = indices2[i]
            if (cnode == head_node):
                continue
            elif (predecessors[cnode] == NULL_IDX):
                node_list[i_nl_end] = cnode
                predecessors[cnode] = pnode
                i_nl_end += 1

        i_nl += 1

    return i_nl
