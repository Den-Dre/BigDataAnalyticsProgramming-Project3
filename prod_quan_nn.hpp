/*
 * author: Laurens Devos
 * Copyright BDAP team, DO NOT REDISTRIBUTE
 *
 *******************************************************************************
 *                                                                             *
 *                     DO NOT CHANGE SIGNATURES OF METHODS!                    *
 *             DO NOT CHANGE METHODS IMPLEMENTED IN THIS HEADER!               *
 *   Sections which MAY require modifications indicated with 'TODO' comments   *
 *                                                                             *
 *******************************************************************************
 */

#include <vector>
#include <iostream>

#include "pydata.hpp"

// self-added
#include <tuple>
#include <queue>

#ifndef PROD_QUAN_NN_GUARD
#define PROD_QUAN_NN_GUARD 1

namespace bdap {

    /** DO NOT MODIFY */
    struct Partition {
        /** Index of the first feature used by this parititon */
        int feat_begin;

        /** Index of the last feature (exclusive) used by this paritition */
        int feat_end;

        /**
         * Number of clusters in this paritition, i.e., the number of
         * centroids.
         */
        int nclusters;

        /**
         * This is a pointer to a C-style row-major matrix containing, in its rows,
         * the `nclusters` cluster centers.
         */
        pydata<float> centroids;

        /**
         * The labels of the training examples: labels[i] is the cluster to
         * which training example i is assigned.
         */
        pydata<int> labels;

        Partition() = delete;
        Partition(const Partition&) = delete;
        Partition(Partition&&) = default;
        inline Partition(int fb, int fe, int nc, pydata<float>&& cs, pydata<int>&& ls)
            : feat_begin(fb)
            , feat_end(fe)
            , nclusters(nc)
            , centroids(std::move(cs))
            , labels(std::move(ls)) {}
    };

    // ========================================================================
    //
    // TODO optionally define additional structs or classes here.
    // Implement them in 'prod_quan_nn.cpp'.
    //
    // ========================================================================


    class ProdQuanNN {
        /** The partitions and their cluster centers. */
        std::vector<Partition> partitions_;

        // ========================================================================
        //
        // TODO optionally add additional structures here to speed up your
        // implementation.
        //
        // Use the C++ standard library! E.g. std::vector is great.
        //
        // Do not use the datatype `pydata`. That is a wrapper for Python-managed
        // objects.
        //
        // ========================================================================

    public:
        /**
         * Constructor. Implemented for you in 'prod_quan_nn.cpp'.
         *
         * DO NOT MODIFY SIGNATURE.
         *
         * Modify implementation if you need to initialize auxiliary
         * structures.
         */
        ProdQuanNN(std::vector<Partition>&& partitions);

        /** DO NOT MODIFY */
        inline const Partition& partition(size_t i) const
        { return partitions_[i]; }

        /** DO NOT MODIFY */
        inline size_t npartitions() const
        { return partitions_.size(); }

        /** DO NOT MODIFY */
        inline size_t nclusters(size_t i) const
        { return partitions_[i].nclusters; }

        /** DO NOT MODIFY */
        inline size_t ntrain_examples() const
        { return partitions_.size() > 0 ? partitions_[0].labels.nrows : 0; }

        /** DO NOT MODIFY */
        inline size_t feat_begin(size_t i) const
        { return partitions_[i].feat_begin; }

        /** DO NOT MODIFY */
        inline size_t feat_end(size_t i) const
        { return partitions_[i].feat_end; }

        /** DO NOT MODIFY */
        inline const float *centroid(size_t i, size_t j) const
        { return partitions_[i].centroids.ptr(j, 0); }

        /** DO NOT MODIFY */
        inline const int *labels(size_t i) const
        { return partitions_[i].labels.ptr(); }

        /**
         * Use this method to initialize your auxiliary structures.
         * This method is called once for you, before any call to
         * `compute_nearest_neighbors` is made.
         *
         * Implement in 'prod_quan_nn.cpp'.
         *
         * DO NOT MODIFY SIGNATURE.
         */
        void initialize_method();

        /**
         * Compute the `nneighbors` nearest neighbors of the given examples.
         *
         * The function has two output arguments, `out_index` and
         * `out_distance`. Both are 2-d row-major C arrays with the same number
         * of rows as `examples`, and `nneighbors` columns. You should write to
         * `out_index` the indices of the nearest neighbors. You should write
         * to `out_distance` the Euclidian distances to the respective nearest
         * neighbors.
         *
         * Implement in 'prod_quan_nn.cpp'.
         *
         * DO NOT MODIFY SIGNATURE.
         */
        void compute_nearest_neighbors(
                const pydata<float>& examples,
                int nneighbors,
                pydata<int>& out_index,
                pydata<float>& out_distance) const;

        // ========================================================================
        //
        // TODO optionally add additional methods here to structure your code 
        //
        // Implement them in 'prod_quan_nn.cpp'
        //
        // ========================================================================

        /**
         * Calculate the L2 norm (Euclidean distance) between the given `example` and `centroid`
         * based on their features in `partition`.
         *
         * @param example The example from which to calculate the distance
         * @param partition The partiton to which `centroid` belongs
         * @param centroid The centroid form which to calculate the distance
         * @return distance: The distance between `example` and `centroid` based on `partition`'s features
         */
        static float distanceToCentroid(const float *example, const Partition &partition, const float* centroid);

        /**
         * Writes to the vector `distances`, s.t. distances[p][c] is the distance of
         * training example `example` to the centroid of cluster `c` in partition `p`
         *
         * @param example: the test example from which the distances to the centroids will be calculated
         * @param distances: the output argument to which the distances will be written
         */
        void calculateDistancesToCentroids(const float* example,
                                           std::vector<std::vector<float>>& distances) const;

        /**
         * Updates Priority Queue `queue` to contain the approximate `k` nearest neighbors' sorted distances
         * for each training example in `examples`
         *
         * @param examples: the training examples
         * @param distancesToCentroids: the distances to each centroid of each partition from the current test example
         * @param queue: output argument priority queue to which the nearest `k` neighbors' sorted distances will be written
         * @param k: the number of neighbors, i.e.: k from kNN
         */
        void getDistancesToExample(const pydata<float> &examples,
                                   const std::vector<std::vector<float>> &distancesToCentroids,
                                   std::vector<std::tuple<float, int>> &distances,
                                   int k) const;
    };



} /* namespace bdap */

#endif // PROD_QUAN_NN_GUARD
