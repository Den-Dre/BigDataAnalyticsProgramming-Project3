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

    void
    ProdQuanNN::compute_nearest_neighbors(
                const pydata<float>& examples,
                int nneighbors,
                pydata<int>& out_index,
                pydata<float>& out_distance) const
    {

//         distancesToCentroids[p][c] is the distance from the test example of this iteration to cluster `c` of partition `p`
//        vector<vector<float>> distancesToCentroids(this->npartitions(), vector<float>(this->nclusters(0)));
//
        // Algorithm of slides: slower than implementation below
//        for (size_t t = 0; t < examples.nrows; t++) {
//            const float* example = examples.ptr(t, 0);
//            // distancesToExample[e] is the approximate distance of the current test example to train example `e`
//            vector<tuple<float, int>> distancesToExample(this->ntrain_examples());
//            for (size_t u = 0; u < this->npartitions(); u++) {
//                vector<float> lookup(this->nclusters(u), 0.0f);
//                const Partition& partition = this->partition(u);
//                for (size_t i = 0; i < this->nclusters(u); i++) {
//                    const float* centroid = this->centroid(u, i);
//                    lookup[i] = distanceToCentroid(example, partition, centroid);
//                }
//                for (size_t x = 0; x < this->ntrain_examples(); x++) {
//                    float newDist = get<0>(distancesToExample[x]) + lookup[this->labels(u)[x]];
//                    distancesToExample[x] = make_tuple(newDist, x);
//                }
//            }
//
//            // Sort the distances to find the `nneighbours` nearest neighbours
//            // Sorted on first element of each tuple, i.e.: the distance
//            sort(distancesToExample.begin(), distancesToExample.end());
////            printTupleVector(distancesToExample);
//
//            // Update the output pointers
//            for (int j = 0; j < nneighbors; j++) {
//                *out_distance.ptr_mut(t, j) = get<0>(distancesToExample.at(j));
//                *out_index.ptr_mut(t, j) = get<1>(distancesToExample.at(j));
//            }
//        }

        // Data structures initialisation

        // distancesToCentroids[p][c] is the distance from the test example of this iteration to cluster `c` of partition `p`
        vector<vector<float>> distancesToCentroids(this->npartitions(), vector<float>(this->nclusters(0)));

        // distancesToExample[e] is the approximate distance of the current test example to train example `e`
        vector<tuple<float, int>> distancesToExample(this->ntrain_examples());

//      For each test example
        for (size_t i = 0; i < examples.nrows; i++) {
            const float *example = examples.ptr(i, 0);
            // https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/
            // "First, we’re going to calculate the squared L2 distance between each subsection of
            // our vector and each of the `nclusters` centroids for that subsection."
            // Get the distances of `example` to each centroid within each of its partitions
            calculateDistancesToCentroids(examples, example, distancesToCentroids);

            // "Remember that each database vector is now just a sequence of `npartitions` centroid ids.
            // To calculate the approximate distance between a given database vector and the query vector,
            // we just use those centroid ids (labels in our case) to look up the partial distances
            // in the table, and sum those up!"
//            getDistancesToExample(examples, distancesToCentroids, distancesToExample);
            priority_queue<tuple<float, int>> queue;
            getDistancesToExample2(examples, distancesToCentroids, queue, nneighbors);

            // Sort the distances to find the `nneighbours` nearest neighbours
            // Sorted on first element of each tuple, i.e.: the distance
            // TODO Use priority queue to prevent sorting all entries
//            sort(distancesToExample.begin(), distancesToExample.end());
//            printTupleVector(distancesToExample);

            // Update the output pointers
//            for (int j = 0; j < nneighbors; j++) {
//                *out_distance.ptr_mut(i, j) = get<0>(distancesToExample.at(j));
//                *out_index.ptr_mut(i, j) = get<1>(distancesToExample.at(j));
//            }
            for (int j = nneighbors-1; j >= 0; j--) {
                tuple<float, int> top = queue.top();
                *out_distance.ptr_mut(i, j) = get<0>(top);
                *out_index.ptr_mut(i, j) = get<1>(top);
                queue.pop();
            }
        }
    }

    // Calculate the squared L2 norm between the given `example` and `centroid`
    float ProdQuanNN::distanceToCentroid(const float* example, const Partition& partition, const float* centroid) {
        float distance = 0.0f;
        for (int fIdx = partition.feat_begin; fIdx < partition.feat_end; fIdx++) {
            distance += powf(example[fIdx] - centroid[fIdx-partition.feat_begin], 2);
        }
        return distance;
    }

    // Returns a vector `distances`, where distances[p][c] is the distance of
    // training example `example` to cluster `c` in partition `p`
    void ProdQuanNN::calculateDistancesToCentroids(const pydata<float>& examples,
                                                   const float* example,
                                                   std::vector<std::vector<float>>& distances) const {
        for (size_t p = 0; p < this->npartitions(); p++) {
            const Partition& partition = this->partition(p);
            for (int c = 0; c < partition.nclusters; c++) {
//                const float* centroid = partition.centroids.ptr(c, 0);
                const float* centroid = this->centroid(p, c);
                distances.at(p).at(c) = distanceToCentroid(example, partition, centroid);
            }
        }
    }

    void ProdQuanNN::getDistancesToExample(const pydata<float> &examples,
                                           const std::vector<std::vector<float>> &distancesToCentroids,
                                           std::vector<std::tuple<float, int>> &distances) const {
        priority_queue<tuple<float, int>> queue;
        // For each training example...
        for (size_t t = 0; t < this->ntrain_examples(); t++) {
           float distanceAcc = 0.0f;
           for (size_t p = 0; p < this->npartitions(); p++) {
               int closestCentroid = this->labels(p)[t];
               // Use the `at` operator to enforce bounds checking:
               distanceAcc += distancesToCentroids.at(p).at(closestCentroid);
           }
           // Store the Euclidean distance (!= squared Euclidean distance)
           distances.at(t) = make_tuple(sqrt(distanceAcc), t);
        }
    }

    void ProdQuanNN::getDistancesToExample2(const pydata<float> &examples,
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
            queue.push(make_tuple(sqrt(distanceAcc), t));
            if (queue.size() > (size_t) k) {
                queue.pop();
            }
        }
    }

    // Utilitarian and debugging methods

    void ProdQuanNN::print2DVector(const std::vector<std::vector<double>>& distances) {
        for (auto & distance : distances) {
            for (double value : distance) {
                cout << value << ", ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    void ProdQuanNN::printTupleVector(const std::vector<std::tuple<float, int>>& distances) {
        for (auto & tup : distances) {
            cout << "<" << get<0>(tup) << ", " << get<1>(tup) << "> ";
        }
        cout << "\n";
    }

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
