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
     *    in `functions.py`
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
            // Get the distances of `example` to each centroid within each of its partitions
            calculateDistancesToCentroids(example, distancesToCentroids);

            // "Remember that each database vector is now just a sequence of `npartitions` centroid ids.
            // To calculate the approximate distance between a given database vector and the query vector,
            // we just use those centroid ids (labels in our case) to look up the partial distances
            // in the table, and sum those up!"
            priority_queue<tuple<float, int>> queue;
            getDistancesToExample(examples, distancesToCentroids, queue, nneighbors);

            // Update the output pointers by sequentially popping the
            // top `nneighbors` entries of the priority queue
            for (int j = nneighbors-1; j >= 0; j--) {
                tuple<float, int> top = queue.top();
                // Store the Euclidean distances, not the squared Euclidean distances
                *out_distance.ptr_mut(i, j) = sqrt(get<0>(top));
                *out_index.ptr_mut(i, j) = get<1>(top);
                queue.pop();
            }
        }
    }

    /**
     * Calculate the L2 norm (Euclidean distance) between the given `example` and `centroid`
     * based on their features in `partition`.
     *
     */
    float ProdQuanNN::distanceToCentroid(const float* example, const Partition& partition, const float* centroid) {
        float distance = 0.0f;
        for (int fIdx = partition.feat_begin; fIdx < partition.feat_end; fIdx++) {
            distance += powf(example[fIdx] - centroid[fIdx-partition.feat_begin], 2);
        }
//        return sqrt(distance);
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
                                           std::priority_queue<std::tuple<float, int>>& queue,
                                           const int k) const {
        // For each training example...
        for (size_t t = 0; t < this->ntrain_examples(); t++) {
            float distanceAcc = 0.0f;
            for (size_t p = 0; p < this->npartitions(); p++) {
                int closestCentroid = this->labels(p)[t];
                // Use the `at` operator to enforce bounds checking:
                distanceAcc += distancesToCentroids.at(p).at(closestCentroid);
            }
            // Store the Euclidean distance (!= squared Euclidean distance)
            queue.push(make_tuple(distanceAcc, t));
            if (queue.size() > (size_t) k) {
                queue.pop();
            }
        }
    }

    // ----------------------------------
    // Utilitarian and debugging methods
    // ----------------------------------

    // Print a nested `vector` in human readable form to `cout`
    void ProdQuanNN::print2DVector(const std::vector<std::vector<double>>& distances) {
        for (auto & distance : distances) {
            for (double value : distance) {
                cout << value << ", ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    // Print a `vector` containing `tuple`s to `cout` in human readable form
    void ProdQuanNN::printTupleVector(const std::vector<std::tuple<float, int>>& distances) {
        for (auto & tup : distances) {
            cout << "<" << get<0>(tup) << ", " << get<1>(tup) << "> ";
        }
        cout << "\n";
    }

    // Print a `vector` to `cout` in human radable form
    void ProdQuanNN::print_vector(const float *ptr, const pydata<float>& examples, const int nneighbors) {
        std::cout << "Compute the " << nneighbors << " nearest neighbors for the "
                  << examples.nrows
                  << " given examples." << std::endl

                  << "The examples are given in C-style row-major order, that is," << std::endl
                  << "the values of a row are consecutive in memory." << std::endl

                  << "The 5th example can be fetched as follows:" << std::endl;
        std::cout << '[';
        for (size_t i = 0; i < examples.ncols; ++i) {
            if (i>0) std::cout << ",";
            if (i>0 && i%5==0) std::cout << std::endl << ' ';
            printf("%11f", ptr[i]);
        }
        std::cout << " ]" << std::endl;
    }
} // namespace bdap
