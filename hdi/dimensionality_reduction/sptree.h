/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#ifndef SPTREE_H
#define SPTREE_H

#include <iostream>
#include <vector>
#include <unordered_map>
#include <map>

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

namespace hdi{
    namespace dr{

        //! Sparse Partitioning Tree used for the Barnes Hut approximation
        /*!
            Sparse Partitioning Tree used for the Barnes Hut approximation.
            The original version was implemented by Laurens van der Maaten,
            \author Laurens van der Maaten
            \author Nicola Pezzotti
        */
        template <typename scalar_type>
        class SPTree{
        public:

            typedef double hp_scalar_type;

        private:
            class Cell {
                unsigned int _emb_dimension;
                hp_scalar_type* corner;
                hp_scalar_type* width;

            public:
                Cell(unsigned int inp__emb_dimension);
                Cell(unsigned int inp__emb_dimension, hp_scalar_type* inp_corner, hp_scalar_type* inp_width);
                ~Cell();

                hp_scalar_type getCorner(unsigned int d);
                hp_scalar_type getWidth(unsigned int d);
                void setCorner(unsigned int d, hp_scalar_type val);
                void setWidth(unsigned int d, hp_scalar_type val);
                bool containsPoint(scalar_type point[]);
            };

            // Fixed constants
            static const unsigned int QT_NODE_CAPACITY = 1;

            // A buffer we use when doing force computations
            //hp_scalar_type* buff;

            // Properties of this node in the tree
            SPTree* parent;
			// Dimensionality of the embedding
            unsigned int _emb_dimension;
            bool is_leaf;
            unsigned int size;
            unsigned int cum_size;
			
			// Cumulative weight of points in this cell
			hp_scalar_type cum_rep_weight;
			//scalar_type* attr_weights; // Reference to weights of the data points
			scalar_type* rep_weights; // Reference to weights of the data points
			scalar_type sum_all_rep_weights; // Sum of all the point weights (used for normalisation)

            // Axis-aligned bounding box stored as a center with half-_emb_dimensions to represent the boundaries of this quad tree
            Cell* boundary;

            // Indices in this space-partitioning tree node, corresponding center-of-mass, and list of all children
            scalar_type* _emb_positions;
            hp_scalar_type* _center_of_mass;
            unsigned int index[QT_NODE_CAPACITY];

            // Children
            SPTree** children;
            unsigned int no_children;

        public:
            //SPTree(unsigned int D, scalar_type* inp_data, unsigned int N);
			SPTree(unsigned int D, scalar_type* inp_data, unsigned int N, scalar_type* point_weights);
        private:
            SPTree(unsigned int D, scalar_type* inp_data, hp_scalar_type* inp_corner, hp_scalar_type* inp_width);
            SPTree(unsigned int D, scalar_type* inp_data, unsigned int N, hp_scalar_type* inp_corner, hp_scalar_type* inp_width);
            SPTree(SPTree* inp_parent, unsigned int D, scalar_type* inp_data, unsigned int N, hp_scalar_type* inp_corner, hp_scalar_type* inp_width);
            SPTree(SPTree* inp_parent, unsigned int D, scalar_type* inp_data, hp_scalar_type* inp_corner, hp_scalar_type* inp_width);
        public:
            ~SPTree();
            void setData(scalar_type* inp_data);
            SPTree* getParent();
            void construct(Cell boundary);
            bool insert(unsigned int new_index);
            void subdivide();
            bool isCorrect();
            void rebuildTree();
            void getAllIndices(unsigned int* indices);
            unsigned int getDepth();
            void computeNonEdgeForcesOMP(unsigned int point_index, hp_scalar_type theta, hp_scalar_type neg_f[], hp_scalar_type& sum_Q)const;
            void computeNonEdgeForces(unsigned int point_index, hp_scalar_type theta, hp_scalar_type neg_f[], hp_scalar_type* sum_Q)const;
            void computeEdgeForces(unsigned int* row_P, unsigned int* col_P, hp_scalar_type* val_P, hp_scalar_type sum_P, int N, hp_scalar_type* pos_f)const;

            template <typename sparse_scalar_matrix>
            void computeEdgeForces(const sparse_scalar_matrix& matrix, hp_scalar_type multiplier, hp_scalar_type* pos_f, scalar_type* attr_weights)const;

            void print();

        private:
            void init(SPTree* inp_parent, unsigned int D, scalar_type* inp_data, hp_scalar_type* inp_corner, hp_scalar_type* inp_width);
            void fill(unsigned int N);
            unsigned int getAllIndices(unsigned int* indices, unsigned int loc);
            bool isChild(unsigned int test_index, unsigned int start, unsigned int end);
        };


/////////////////////////////////////////////////////////////////////////

        template <typename scalar_type>
        template <typename sparse_scalar_matrix>
		// Positive Forces / Attractive Forces / F_attr
		// sparse_scalar_matrix = std::vector<hdi::data::MapMemEff<uint32_t,float>>>, sparse_scalar_matrix is a list of lists containing int->float pairs
        void SPTree<scalar_type>::computeEdgeForces(const sparse_scalar_matrix& sparse_matrix, hp_scalar_type exaggeration, hp_scalar_type* pos_f, scalar_type* attr_weights)const{
			const int n = sparse_matrix.size(); // n is the total number of data points

			//float sum_weights = 0;
			//for (int i = 0; i < n; i++) sum_weights += attr_weights[i];

            // Loop over all edges in the graph
#ifdef __APPLE__
            //std::cout << "GCD dispatch, sptree 180.\n";
            dispatch_apply(n, dispatch_get_global_queue(0, 0), ^(size_t j) {
#else
#pragma omp parallel for
			// Loop over all data points
            for(int j = 0; j < n; ++j) {
#endif //__APPLE__
				// Allocate buffer for distance computation
                std::vector<hp_scalar_type> buff(_emb_dimension,0);

				// Index of Yi
				unsigned int ind1;

				// Index of Yj
				unsigned int ind2;

                hp_scalar_type D;

				// Index of the first coordinate of Yi
                ind1 = j * _emb_dimension;

				// Loop over all non-zero connections from data point j
                for(auto elem: sparse_matrix[j]) { // j is the index of i, elem.first is the index of j, elem.second/n is the value p_ij
                    // Compute pairwise distance and Q-value
                    D = 1.0; // q_ij = 1 + ||Yi - Yj||^2
					
                    ind2 = elem.first * _emb_dimension; // Index of the first coordinate of Yj
                    
					// Compute squared euclidean distance between Yi and Yj
					for(unsigned int d = 0; d < _emb_dimension; d++)
                        buff[d] = _emb_positions[ind1 + d] - _emb_positions[ind2 + d]; // buff contains (yi-yj) for each _emb_dimension
                    
					for(unsigned int d = 0; d < _emb_dimension; d++)
                        D += buff[d] * buff[d];

                    hp_scalar_type p_ij = elem.second / n;
                    //hp_scalar_type res = hp_scalar_type(p_ij) * exaggeration / D
					hp_scalar_type res = hp_scalar_type(p_ij) * exaggeration / D; // p_ij / (1 + ||y_i - y_j||^2)

					// Fetch the weight value for this connection (average weight)
					//float weight = 0.5f * (attr_weights[j] + attr_weights[elem.first]);
					float weight = attr_weights[elem.first];// / (sum_weights - attr_weights[j]);

                    // Add the positive force to the existing force for every dimension in the embedding
                    for(unsigned int d = 0; d < _emb_dimension; d++)
                      pos_f[ind1 + d] += weight * res * buff[d]; // (p_ij * (y_i - y_j)) / (1 + ||y_i - y_j||^2)
                }
            }
#ifdef __APPLE__
            );
#endif
        }

    }
}
#endif
