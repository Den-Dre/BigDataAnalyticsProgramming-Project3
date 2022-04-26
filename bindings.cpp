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

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/ndarrayobject.h>

#include <iostream>
#include <vector>

#include "pydata.hpp"
#include "prod_quan_nn.hpp"

using bdap::pydata;

// https://docs.python.org/3/c-api/typeobj.html
// https://docs.python.org/3/extending/newtypes_tutorial.html#adding-data-and-methods-to-the-basic-example

static bool
validate_data(PyObject* arr, int ndim, int dtype)
{
#define VD_ERR(msg) {\
    PyErr_SetString(PyExc_ValueError, msg);\
    Py_DECREF(arr);\
    return false; }\

    if (!PyArray_Check(arr)) VD_ERR("not a numpy array");
    if (PyArray_NDIM(arr) != ndim) VD_ERR("need 2-dim ndarray")
    if (!PyArray_ISCARRAY(arr)) VD_ERR("invalid array format");
    if (PyArray_TYPE(arr) != dtype) VD_ERR("expected float32");

    return true;
#undef VD_ERR
}


template <typename T>
static pydata<T>
get_data(PyArrayObject *arr)
{
    typename pydata<T>::data_ptr ptr{
        (T *) PyArray_DATA(arr),
        [arr](T *) { Py_DECREF(arr); }};

    pydata<T> d(std::move(ptr), 0, 0, 0, 0);
    int ndim = PyArray_NDIM(arr);
    d.nrows = PyArray_DIM(arr, 0);
    d.ncols = ndim == 2 ? PyArray_DIM(arr, 1) : 1;
    d.stride_row = PyArray_STRIDE(arr, 0) / sizeof(T);
    d.stride_col = ndim == 2 ? PyArray_STRIDE(arr, 1) / sizeof(T) : 1;

    return d;
}

extern "C" {
    typedef struct {
        PyObject_HEAD
        bdap::ProdQuanNN pqnn;
    } PyProdQuanNNObject;

    PyObject *myobj_new(PyTypeObject *subtype, PyObject *args, PyObject *kwds);

    static void
    PyProdQuanNNDealloc(PyProdQuanNNObject *self)
    {
        {
            bdap::ProdQuanNN pqnn = std::move(self->pqnn);
            // destruct
        }
        Py_TYPE(self)->tp_free((PyObject *) self);
    }

    static PyObject *
    PyProdQuanNNNew(PyTypeObject *type, PyObject *args, PyObject *kwds)
    {
        PyProdQuanNNObject *self;
        self = (PyProdQuanNNObject *) type->tp_alloc(type, 0);
        if (self != NULL) {
            new (&self->pqnn) bdap::ProdQuanNN({});
        }
        return (PyObject *) self;
    }

    static int
    PyProdQuanNNInit(PyProdQuanNNObject *self, PyObject *args, PyObject *kwds)
    {
        PyObject *pypartitions;

        if (!PyArg_ParseTuple(args, "O", &pypartitions))
            return -1;

        Py_INCREF(pypartitions);

        if (!PyList_Check(pypartitions)) {
            Py_DECREF(pypartitions);
            PyErr_SetString(PyExc_TypeError, "partitions must be a list");
            return -1;
        }

        int npartitions = PyList_Size(pypartitions);
        if (npartitions < 0) {
            Py_DECREF(pypartitions);
            return -1;
        }

        std::vector<bdap::Partition> partitions;

        for (int i = 0; i < npartitions; ++i)
        {
            PyObject *o = PyList_GetItem(pypartitions, i); // borrow
            if (!PyDict_Check(o)) {
                Py_DECREF(pypartitions);
                PyErr_SetString(PyExc_ValueError, "partitions' items must be dicts");
                return -1;
            }

            PyObject *begin = PyDict_GetItemString(o, "begin"); // borrow
            PyObject *end = PyDict_GetItemString(o, "end"); // borrow
            PyObject *kmeans = PyDict_GetItemString(o, "kmeans"); // borrow

            if (!begin || !end || !kmeans) {
                Py_DECREF(pypartitions);
                PyErr_SetString(PyExc_ValueError, "begin, end or kmeans NULL");
                return -1;
            }

            int b = (int) PyLong_AsLong(begin);
            int e = (int) PyLong_AsLong(end);

            if (PyErr_Occurred()) {
                Py_DECREF(pypartitions);
                return -1;
            }

            PyObject *centroids = PyObject_GetAttrString(kmeans, "cluster_centers_");
            if (!centroids || !validate_data(centroids, 2, NPY_FLOAT32)) {
                Py_DECREF(pypartitions);
                Py_XDECREF(centroids);
                return -1;
            }

            PyObject *labels = PyObject_GetAttrString(kmeans, "labels_");
            if (!labels || !validate_data(labels, 1, NPY_INT32)) {
                Py_DECREF(pypartitions);
                Py_DECREF(centroids);
                Py_XDECREF(centroids);
                return -1;
            }

            // get_data deleter DECREFs centroids and labels when data goes out
            // of scope
            pydata<float> d = get_data<float>((PyArrayObject *) centroids);
            pydata<int> l = get_data<int>((PyArrayObject *) labels);

            partitions.emplace_back(b, e, static_cast<int>(d.nrows),
                    std::move(d), std::move(l));
        }

        self->pqnn = bdap::ProdQuanNN(std::move(partitions));
        Py_DECREF(pypartitions);
        return 0;
    }

    static PyObject *
    ProdQuanNN_npartitions(PyProdQuanNNObject *self,  PyObject *Py_UNUSED(ignored))
    { return PyLong_FromSize_t(self->pqnn.npartitions()); }

    int
    is_valid_partition_index(PyProdQuanNNObject *self, PyObject *pyi)
    {
        int i = (int) PyLong_AsLong(pyi);
        if (PyErr_Occurred()) return -1;
        if (i < 0 || i >= (int)self->pqnn.npartitions()) \
        {
            PyErr_SetString(PyExc_IndexError, "partition index out of bounds");
            return -1;
        }
        return i;
    }

    int
    is_valid_cluster_index(PyProdQuanNNObject *self, int i, PyObject *pyj)
    {
        int j = (int) PyLong_AsLong(pyj);
        if (PyErr_Occurred()) return -1;
        if (j < 0 || j >= (int)self->pqnn.nclusters(i)) \
        {
            PyErr_SetString(PyExc_IndexError, "cluster index out of bounds");
            return -1;
        }
        return j;
    }

    static PyObject *
    ProdQuanNN_nclusters(PyProdQuanNNObject *self, PyObject *pyi)
    {
        int i = -1;
        if ((i = is_valid_partition_index(self, pyi)) != -1)
            return PyLong_FromSize_t(self->pqnn.nclusters(i));
        return NULL;
    }

    static PyObject *
    ProdQuanNN_nexamples(PyProdQuanNNObject *self, PyObject *Py_UNUSED(ignored))
    { return PyLong_FromSize_t(self->pqnn.ntrain_examples()); }

    static PyObject *
    ProdQuanNN_feat_begin(PyProdQuanNNObject *self, PyObject *pyi)
    {
        int i = -1;
        if ((i = is_valid_partition_index(self, pyi)) != -1)
            return PyLong_FromSize_t(self->pqnn.feat_begin(i));
        return NULL;
    }

    static PyObject *
    ProdQuanNN_feat_end(PyProdQuanNNObject *self, PyObject *pyi)
    {
        int i = -1;
        if ((i = is_valid_partition_index(self, pyi)) != -1)
            return PyLong_FromSize_t(self->pqnn.feat_end(i));
        return NULL;
    }

    static PyObject *
    ProdQuanNN_centroid(PyProdQuanNNObject *self, PyObject *args)
    {
        PyObject *pyi = NULL, *pyj = NULL;
        if (!PyArg_ParseTuple(args, "OO", &pyi, &pyj))
            return NULL;
        int i = is_valid_partition_index(self, pyi);
        if (i == -1) return NULL;
        int j = is_valid_cluster_index(self, i, pyj);
        if (j == -1) return NULL;

        npy_long sz = self->pqnn.partition(i).centroids.ncols;
        PyObject *arr = PyArray_SimpleNew(1, &sz, NPY_FLOAT32);
        if (arr == NULL) { return PyErr_NoMemory(); }
        float *dst = (float *) PyArray_DATA(arr);
        float const *src = self->pqnn.centroid(i, j);
        std::copy(src, src+sz, dst);
        return arr;
    }

    static PyObject *
    ProdQuanNN_labels(PyProdQuanNNObject *self, PyObject *pyi)
    {
        int i = is_valid_partition_index(self, pyi);
        if (i == -1) return NULL;

        npy_long nex = self->pqnn.ntrain_examples();
        if (nex == 0) {
            PyErr_SetString(PyExc_ValueError, "Empty ProdQuanNN");
            return NULL;
        }

        PyObject *arr = PyArray_SimpleNew(1, &nex, NPY_INT32);
        if (arr == NULL) { return PyErr_NoMemory(); }
        int *dst = (int *) PyArray_DATA(arr);
        int const *src = self->pqnn.labels(i);
        std::copy(src, src+nex, dst);
        return arr;
    }

    static PyObject *
    ProdQuanNN_initialize_method(PyProdQuanNNObject *self,
            PyObject *Py_UNUSED(ignored))
    {
        self->pqnn.initialize_method();
        return Py_None;
    }

    static PyObject *
    ProdQuanNN_compute_nearest_neighbors(PyProdQuanNNObject *self,
            PyObject *args)
    {
        PyObject *pyexamples = NULL;
        int k;

        if (!PyArg_ParseTuple(args, "Oi", &pyexamples, &k))
            return NULL;

        Py_INCREF(pyexamples);

        if (!validate_data(pyexamples, 2, NPY_FLOAT32)) {
            Py_DECREF(pyexamples);
            return NULL;
        }

        auto examples = get_data<float>((PyArrayObject *) pyexamples);

        npy_long dims[2] {(npy_long) examples.nrows, k};
        PyObject *arr_index = PyArray_ZEROS(2, dims, NPY_INT32, 0);
        if (arr_index == NULL) { return PyErr_NoMemory(); }
        PyObject *arr_distance = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
        if (arr_distance == NULL) { Py_DECREF(arr_index); return PyErr_NoMemory(); }

        {
            Py_INCREF(arr_index); // pydata<int> DECREFs again!
            pydata<int> out_index = get_data<int>((PyArrayObject *) arr_index);
            Py_INCREF(arr_distance); // pydata<int> DECREFs again!
            pydata<float> out_distance = get_data<float>((PyArrayObject *) arr_distance);
            self->pqnn.compute_nearest_neighbors(examples, k, out_index, out_distance);
        }

        return Py_BuildValue("NN", arr_index, arr_distance);
    }

    static PyMethodDef ProdQuanNN_methods[] = {
        {"npartitions", (PyCFunction) ProdQuanNN_npartitions, METH_NOARGS,
         "Get the number of feature partitions" },
        {"nclusters", (PyCFunction) ProdQuanNN_nclusters, METH_O,
         "Get the number of clusters in the ith partition" },
        {"ntrain_examples", (PyCFunction) ProdQuanNN_nexamples, METH_NOARGS,
         "Get the number training set examples" },
        {"feat_begin", (PyCFunction) ProdQuanNN_feat_begin, METH_O,
         "Index of first feature in partition" },
        {"feat_end", (PyCFunction) ProdQuanNN_feat_end, METH_O,
         "Index of last feature in partition (exclusive)" },
        {"centroid", (PyCFunction) ProdQuanNN_centroid, METH_VARARGS,
         "Get a centroid from a partition" },
        {"labels", (PyCFunction) ProdQuanNN_labels, METH_O,
         "Get the training set labels for a partitions" },
        {"initialize_method", (PyCFunction) ProdQuanNN_initialize_method, METH_NOARGS,
         "Initialize ProdQuanNN" },
        {"compute_nearest_neighbors",
            (PyCFunction) ProdQuanNN_compute_nearest_neighbors, METH_VARARGS,
         "Compute NNs" },
        {NULL}  /* Sentinel */
    };

    static PyTypeObject PyProdQuanNNType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        .tp_name = "prod_quan_nn.ProdQuanNN",
        .tp_basicsize = sizeof(PyProdQuanNNObject),
        .tp_dealloc = (destructor)PyProdQuanNNDealloc,
        .tp_doc = "ProdQuanNN C++ type",
        .tp_methods = ProdQuanNN_methods,
        .tp_init = (initproc) PyProdQuanNNInit,
        .tp_new = PyProdQuanNNNew,
        //.tp_repr = (reprfunc)myobj_repr,
    };

    static PyMethodDef ProdQuanNN_static[] = {
        //{"constructProdQuanNN",  prod_quan_nn_construct, METH_O, ""},
        {NULL, NULL, 0, NULL}        /* Sentinel */
    };

    static struct PyModuleDef prod_quan_nn_module = {
        PyModuleDef_HEAD_INIT,
        "prod_quan_nn",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
        ProdQuanNN_static
    };

    PyMODINIT_FUNC
    PyInit_prod_quan_nn(void)
    {
        PyObject *m;
        // initialize numpy
        // https://numpy.org/doc/stable/reference/c-api/array.html?highlight=import%20array#importing-the-api
        import_array();

        if (PyType_Ready(&PyProdQuanNNType) < 0)
            return NULL;

        m = PyModule_Create(&prod_quan_nn_module);
        if (m == NULL) return NULL;

        Py_INCREF(&PyProdQuanNNType);
        if (PyModule_AddObject(m, "ProdQuanNN", (PyObject *) &PyProdQuanNNType) < 0) {
            Py_DECREF(&PyProdQuanNNType);
            Py_DECREF(m);
            return NULL;
        }
        return m;
    }
} /* extern "C" */
