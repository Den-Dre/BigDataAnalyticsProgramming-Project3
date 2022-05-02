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
    bool sortbysec(const tuple<float, int>& a,
                   const tuple<float, int>& b)
    {
        return (get<1>(a) < get<1>(b));
    }

    void
    ProdQuanNN::compute_nearest_neighbors(
                const pydata<float>& examples,
                int nneighbors,
                pydata<int>& out_index,
                pydata<float>& out_distance) const
    {
        std::cout << "Compute the " << nneighbors << " nearest neighbors for the "
            << examples.nrows
            << " given examples." << std::endl

            << "The examples are given in C-style row-major order, that is," << std::endl
            << "the values of a row are consecutive in memory." << std::endl

            << "The 5th example can be fetched as follows:" << std::endl;

        print_vector(examples.ptr(5, 0), examples.ncols);

        // TODO use array?

        // TODO convert to class members?
        const float *example;
        // -1 As we exclude to distance from a vector to itself:
        vector<tuple<float, int>> distancesToExample(examples.nrows-1);
        vector<vector<float>> distancesToCentroids(this->npartitions(), vector<float>(this->nclusters(0)));

        // For each example
        for (size_t i = 0; i < examples.nrows; i++) {
            example = examples.ptr(i, 0);
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
            getDistancesToExample(examples, i, distancesToCentroids, distancesToExample);

            // TODO use a priority queue instead of a vector?
            // Sort the distances to find the `nneighbours` nearest neighbours
            // Sorted on first element of each tuple, i.e.: the distance
            sort(distancesToExample.begin(), distancesToExample.end());

            // Update the output pointers
            for (int j = 0; j < nneighbors; j++) {
                *out_distance.ptr_mut(i, j) = get<0>(distancesToExample[j]);
                *out_index.ptr_mut(i, j) = get<1>(distancesToExample[j]);
            }
            cout << nneighbors << " Nearest neighbours of [" << *example <<", ...]: \t";
            for (int idx = 0; idx < nneighbors; idx++)
                cout << "<" << get<0>(distancesToExample[idx])
                        << ", " << get<1>(distancesToExample[idx]) << ">";
            cout << endl;
//            cout << std::endl << "Sorted points:" << std::endl;
//            sort(distancesToExample.begin(), distancesToExample.end(), sortbysec);
//            for (auto val: distancesToExample)
//                cout << "<" << get<0>(val) << ", " << get<1>(val) << ">" << std::endl;
//            break;
        }

    }

    float ProdQuanNN::distanceToCentroid(const float *example, const Partition& partition, size_t cIdx) {
        float distance = 0;
        const float* centroid = partition.centroids.ptr(cIdx, 0);
        for (int fIdx = partition.feat_begin; fIdx < partition.feat_end; fIdx++) {
            distance += (float) pow(example[fIdx] - centroid[fIdx-partition.feat_begin], 2);
        }
        return distance;
    }

    void ProdQuanNN::calculateDistancesToCentroids(const pydata<float>& examples, const float* example, std::vector<std::vector<float>>& distances) const {
        // distances[p][c] is the distance from `example` to cluster `c` in partition `p`
        // TODO use reference move with && or something?

        for (size_t p = 0; p < this->npartitions(); p++) {
            const Partition& partition = this->partition(p);
            for (int c = 0; c < partition.nclusters; c++) {
                distances.at(p).at(c) = distanceToCentroid(example, partition, c);
            }
        }
    }

    void ProdQuanNN::getDistancesToExample(const pydata<float>& examples, size_t i, const std::vector<std::vector<float>>& distancesToCentroids, std::vector<std::tuple<float, int>>& distances) const {
        float distanceAcc;
        int closestCentroid;

        assert(examples.nrows - 1 == distances.size());
        // For each example...
        for (size_t e = 0; e < examples.nrows; e++) {
            if (e == i)
                // Don't calculate distance to itself
                continue;
            distanceAcc = 0;
            for (size_t p = 0; p < this->npartitions(); p++) {
                closestCentroid = this->labels(p)[e];
                // Use the `at` operator to enforce bounds checking:
                distanceAcc += distancesToCentroids.at(p).at(closestCentroid);
            }
            // Offset by one if we have passed index `i`
            distances.at(e > i ? e-1 : e) = make_tuple(distanceAcc, e);
        }
    }

    void ProdQuanNN::print2DVector(const std::vector<std::vector<double>>& distances) {
        for (auto & distance : distances) {
            for (double value : distance) {
                cout << value << ", ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    void ProdQuanNN::print_vector(const float *ptr, size_t ncols) {
        std::cout << '[';
        for (size_t i = 0; i < ncols; ++i) {
            if (i>0) std::cout << ",";
            if (i>0 && i%5==0) std::cout << std::endl << ' ';
            printf("%11f", ptr[i]);
        }
        std::cout << " ]" << std::endl;
    }
} // namespace bdap
