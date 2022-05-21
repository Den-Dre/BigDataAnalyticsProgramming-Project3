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

//        print_vector(examples.ptr(5, 0), examples, nneighbors);

        // TODO use array?
        // TODO convert to class members?

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
            // our vector and each of the 256 centroids for that subsection."
            // Get the distancesToCentroids of `example` to each centroid within each of its partitions
            calculateDistancesToCentroids(examples, example, distancesToCentroids);
//            print2DVector(distancesToCentroids);

            /*
             * Remember that each database vector is now just a sequence of 8 centroid ids.
             * To calculate the approximate distance between a given database vector and the query vector,
             * we just use those centroid ids to look up the partial distancesToCentroids in the table, and sum those up!
             */
            getDistancesToExample(examples, distancesToCentroids, distancesToExample);

            // Sort the distances to find the `nneighbours` nearest neighbours
            // Sorted on first element of each tuple, i.e.: the distance
            sort(distancesToExample.begin(), distancesToExample.end());
//            printTupleVector(distancesToExample);

            // Update the output pointers
            for (int j = 0; j < nneighbors; j++) {
                *out_distance.ptr_mut(i, j) = get<0>(distancesToExample.at(j));
                *out_index.ptr_mut(i, j) = get<1>(distancesToExample.at(j));
            }
        }
    }

    float ProdQuanNN::distanceToCentroid(const float *example, const Partition& partition, const float* centroid) {
        float distance = 0;
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
        // distances[p][c] is the distance from `example` to cluster `c` in partition `p`

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
        // For each training example...
        for (size_t t = 0; t < this->ntrain_examples(); t++) {
           float distanceAcc = 0;
           for (size_t p = 0; p < this->npartitions(); p++) {
               int closestCentroid = this->labels(p)[t];
               // Use the `at` operator to enforce bounds checking:
               distanceAcc += distancesToCentroids.at(p).at(closestCentroid);
           }
           // Store the Euclidean distance ( != squared Euclidean distance)
           distances.at(t) = make_tuple(sqrt(distanceAcc), t);
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
