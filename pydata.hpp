/*
 * author: Laurens Devos
 * Copyright BDAP team, DO NOT REDISTRIBUTE
 *
 *******************************************************************************
 *                                                                             *
 *                          DO NOT MODIFY THIS FILE!                           *
 *                                                                             *
 *******************************************************************************
 */

#include <memory>
#include <functional>

#ifndef PYDATA_GUARD
#define PYDATA_GUARD 1

namespace bdap {

    /**
     * A data pointer wrapper. Data is expected to have a
     * [numpy](https://numpy.org/) layout.
     *
     * - https://docs.python.org/3/c-api/buffer.html#buffer-structure
     * - https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html
     *
     * Do not use this for your own structures. This structure manages data
     * owned by Python.
     *
     * Based on:
     * https://github.com/laudv/veritas/blob/main/src/cpp/basics.hpp#L39
     *
     * DO NOT MODIFY
     */
    template <typename T>
    struct pydata {
        using data_ptr = std::unique_ptr<T, std::function<void(T *)>>;

    private:
        data_ptr ptr_;

    public:
        size_t nrows, ncols;
        size_t stride_row, stride_col; // in num of elems, not bytes


        /** Compute the index of an element. */
        inline size_t index(size_t row, size_t col) const
        { return row * stride_row + col * stride_col; }

        /** Get a pointer to the data */
        inline const T *ptr() const { return ptr_.get(); }

        /** Get a pointer to an element */
        inline const T *ptr(size_t row, size_t col) const
        { return &ptr()[index(row, col)]; }

        /** Get a pointer to the data */
        inline T *ptr_mut() { return ptr_.get(); }

        /** Get a pointer to an element */
        inline T *ptr_mut(size_t row, size_t col)
        { return &ptr_mut()[index(row, col)]; }

        /** Access element in data matrix without bounds checking. */
        inline T get_elem(size_t row, size_t col) const
        { return ptr()[index(row, col)]; }

        /** Access element in data matrix without bounds checking. */
        inline void set_elem(size_t row, size_t col, T&& value)
        { ptr_mut()[index(row, col)] = std::move(value); }

        /** Access elements in the first row (e.g. for when data is vector). */
        inline T operator[](size_t i) const
        { return ptr()[i]; }

        pydata(data_ptr ptr, size_t nr, size_t nc, size_t sr, size_t sc)
            : ptr_(std::move(ptr))
            , nrows(nr)
            , ncols(nc)
            , stride_row(sr)
            , stride_col(sc) {}
    };

} /* namespace bdap */

#endif /* PYDATA_GUARD */

