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
        std::cout << "Compute the nearest neighbors for the "
            << examples.nrows
            << " given examples." << std::endl

            << "The examples are given in C-style row-major order, that is," << std::endl
            << "the values of a row are consecutive in memory." << std::endl

            << "The 5th example can be fetched as follows:" << std::endl;

        print_vector(examples.ptr(5, 0), examples.ncols);

        // TODO use array?
        vector<vector<double>> distances(this->npartitions(), vector<double>(this->nclusters(0)));

        const float *example;
        for (size_t i = 0; i < examples.nrows; i++) {
            // https://mccormickml.com/2017/10/13/product-quantizer-tutorial-part-1/
            // "First, we’re going to calculate the squared L2 distance between each subsection of
            // our vector and each of the 256 centroids for that subsection."
            example = examples.ptr(i, 0);
            calculateDistances(examples, example, distances);
        }

    }

    double ProdQuanNN::distanceToCentroid(const float *example, const Partition& partition, size_t cIdx) {
        double distance = 0;
        const float* centroid = partition.centroids.ptr(cIdx, 0);
        for (int fIdx = partition.feat_begin; fIdx < partition.feat_end; fIdx++) {
            distance += pow(example[fIdx] - centroid[fIdx-partition.feat_begin], 2);
        }
        return distance;
    }

    void ProdQuanNN::calculateDistances(const pydata<float>& examples, const float* example, std::vector<std::vector<double>> distances) const {
        // Each row is a partition, each element in a row is a distance to a cluster
        // TODO use reference move with && or something?

        for (size_t p = 0; p < this->npartitions(); p++) {
            const Partition& partition = this->partition(p);
            for (int c = 0; c < partition.nclusters; c++) {
                distances[p][c] = distanceToCentroid(example, partition, c);
            }
        }
        print2DVector(distances);
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

    void ProdQuanNN::print_vector(const double *ptr, size_t ncols) {
        std::cout << '[';
        for (size_t i = 0; i < ncols; ++i) {
            if (i>0) std::cout << ",";
            if (i>0 && i%5==0) std::cout << std::endl << ' ';
            printf("%11f", ptr[i]);
        }
        std::cout << " ]" << std::endl;
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
