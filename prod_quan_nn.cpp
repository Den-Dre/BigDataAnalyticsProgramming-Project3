/*
 * author: Laurens Devos
 * Copyright BDAP team, DO NOT REDISTRIBUTE
 *
 *******************************************************************************
 *                                                                             *
 *                     DO NOT CHANGE SIGNATURES OF METHODS!                    *
 *             DO NOT CHANGE METHODS IMPLEMENTED IN THIS HEADER!               *
 *     Sections which require modifications indicated with 'TODO' comments     *
 *                                                                             *
 *******************************************************************************
 */

#include "prod_quan_nn.hpp"
#include <limits>
#include <chrono>
#include <cmath>

// Self-added:
#include <bits/stdc++.h>
#include <tuple>
#include <queue>

using namespace std;

namespace bdap {

    // Constructor, modify if necessary for auxiliary structures
    ProdQuanNN::ProdQuanNN(std::vector<Partition>&& partitions)
        : partitions_(std::move(partitions))
    {}

    void
    ProdQuanNN::initialize_method()
    {
        //std::cout << "Construct auxiliary structures here" << std::endl;
    }

    /**
     * Compute the `nneighbors` nearest neighbors of the given examples.
     *
     * It was chosen to store the Euclidean distances rather than the squared Euclidean distances
     * based on the following:
     *  - As sqrt is a monotone function, this doesn't change the relative ordering of the neighbors
     *  - This ensures that the same distance metrics are compared when calculating the Mean Absolute Error
     *    in `functions.py`, rather than comparing for example squared distances to regular distances of sknn
     *  - The extra percentual relative overhead of calculating `nneighbors` square root operations
     *    is negligable, which ensures that this modification doesn't hurt the perofrmance of the
     *    Product Quantization implementation
     */
    void
    ProdQuanNN::compute_nearest_neighbors(
            const pydata<float>& examples,
                int nneighbors,
                pydata<int>& out_index,
                pydata<float>& out_distance) const
    {
        // Data structures initialisation

        // distancesToCentroids[p][c] is the distance from the test example of this iteration to cluster `c` of partition `p`
        vector<vector<float>> distancesToCentroids(this->npartitions(), vector<float>(this->nclusters(0)));

        // distancesToExample[e] is the approximate distance of the current test example to train example `e`
        vector<tuple<float, int>> distancesToExample(this->ntrain_examples());

        // For each test example
        for (size_t i = 0; i < examples.nrows; i++) {
            const float *example = examples.ptr(i, 0);
            // https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/
            // "First, we’re going to calculate the squared L2 distance between each subsection of
            // our vector and each of the `nclusters` centroids for that subsection."
            // Get the distances of `example` to each centroid within each of its partitions:
            calculateDistancesToCentroids(example, distancesToCentroids);

            // https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/
            // "Remember that each database vector is now just a sequence of `npartitions` centroid ids.
            // To calculate the approximate distance between a given database vector and the query vector,
            // we just use those centroid ids (labels in our case) to look up the partial distances
            // in the table, and sum those up!"
            getDistancesToExample(examples, distancesToCentroids, distancesToExample, nneighbors);

            // Only partially sorting `distancesToExamples` suffices to obtain the smallest `nneighbors` entries
            partial_sort(distancesToExample.begin(),
                         distancesToExample.begin() + nneighbors,
                         distancesToExample.end());

            for (int j = 0; j < nneighbors; j++) {
                // Store the Euclidean distances, not the squared Euclidean distances
                *out_distance.ptr_mut(i, j) = sqrt(get<0>(distancesToExample.at(j)));
                *out_index.ptr_mut(i, j) = get<1>(distancesToExample.at(j));
            }
        }
    }

    /**
     * Calculate the L2 norm (Euclidean distance) between the given `example` and `centroid`
     * based on their features in `partition`.
     *
     * @param example The example from which to calculate the distance
     * @param partition The partiton to which `centroid` belongs
     * @param centroid The centroid form which to calculate the distance
     * @return distance: The distance between `example` and `centroid` based on `partition`'s features
     */
    float ProdQuanNN::distanceToCentroid(const float* example, const Partition& partition, const float* centroid) {
        float distance = 0.0f;
        for (int fIdx = partition.feat_begin; fIdx < partition.feat_end; fIdx++) {
            distance += powf(example[fIdx] - centroid[fIdx-partition.feat_begin], 2);
        }
        return distance;
    }

    /**
     * Writes to the vector `distances`, s.t. distances[p][c] is the distance of
     * training example `example` to the centroid of cluster `c` in partition `p`
     *
     * @param example: the test example from which the distances to the centroids will be calculated
     * @param distances: the output argument to which the distances will be written
     */
    void ProdQuanNN::calculateDistancesToCentroids(const float* example,
                                                   std::vector<std::vector<float>>& distances) const {
        for (size_t p = 0; p < this->npartitions(); p++) {
            const Partition& partition = this->partition(p);
            for (int c = 0; c < partition.nclusters; c++) {
                const float* centroid = this->centroid(p, c);
                distances.at(p).at(c) = distanceToCentroid(example, partition, centroid);
            }
        }
    }

    /**
     * Updates Priority Queue `queue` to contain the approximate `k` nearest neighbors' sorted distances
     * for each training example in `examples`
     *
     * @param examples: the training examples
     * @param distancesToCentroids: the distances to each centroid of each partition from the current test example
     * @param queue: output argument priority queue to which the nearest `k` neighbors' sorted distances will be written
     * @param k: the number of neighbors, i.e.: k from kNN
     */
    void ProdQuanNN::getDistancesToExample(const pydata<float> &examples,
                                           const std::vector<std::vector<float>> &distancesToCentroids,
                                           std::vector<std::tuple<float, int>>& res,
                                           const int k) const {
        // For each training example...
        for (size_t t = 0; t < this->ntrain_examples(); t++) {
            float distanceAcc = 0.0f;
            for (size_t p = 0; p < this->npartitions(); p++) {
                int closestCentroid = this->labels(p)[t];
                // Use the `at` operator to enforce bounds checking:
                distanceAcc += distancesToCentroids.at(p).at(closestCentroid);
            }
            res.at(t) = make_tuple(distanceAcc, t);
        }
    }
} // namespace bdap
