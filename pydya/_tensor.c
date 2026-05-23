/*
 * pydya._tensor — C 레벨 1D float32 Tensor primitive.
 *
 * Nadya 의 tensor 가 컴파일러 단의 primitive 데이터 타입이듯, Pydya 도
 * 텐서 연산을 C 레벨에서 수행해 (1) 자동 벡터화(-O3 -march=native 로
 * 컴파일러가 SIMD 깔아 줌)와 (2) 산술 핫루프 GIL 해제를 통한 진짜 멀티코어
 * 잠재력을 확보한다.
 *
 * 1차 범위: 1D float32 contiguous, element-wise add/sub/mul, 스칼라 broadcast,
 * __getitem__/__setitem__/__len__/to_list/from_list/__repr__.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>

/* ─── 정렬된 부동소수점 버퍼 할당 (SIMD 친화) ───────────────────────────── */

static float *alloc_floats(Py_ssize_t n) {
    if (n <= 0) return NULL;
    /* aligned_alloc 는 size 가 alignment 의 배수여야 한다 */
    size_t bytes = ((size_t)n * sizeof(float) + 63u) & ~(size_t)63u;
    void *p = aligned_alloc(64, bytes);
    return (float *)p;
}

/* ─── Tensor 객체 ──────────────────────────────────────────────────────── */

typedef struct {
    PyObject_HEAD
    float *data;
    Py_ssize_t size;
} TensorObject;

static PyTypeObject TensorType;  /* forward */

static void Tensor_dealloc(TensorObject *self) {
    free(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int Tensor_init(TensorObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"data", "fill", NULL};
    PyObject *first = NULL;
    double fill = 0.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d", kwlist, &first, &fill)) {
        return -1;
    }

    /* 정수면 size 로 해석해 fill 값으로 초기화 */
    if (PyLong_CheckExact(first)) {
        Py_ssize_t n = PyLong_AsSsize_t(first);
        if (n == -1 && PyErr_Occurred()) return -1;
        if (n < 0) {
            PyErr_SetString(PyExc_ValueError, "size must be non-negative");
            return -1;
        }
        free(self->data);
        self->size = n;
        self->data = alloc_floats(n);
        if (self->data == NULL && n > 0) {
            PyErr_NoMemory();
            return -1;
        }
        const float fv = (float)fill;
        for (Py_ssize_t i = 0; i < n; ++i) self->data[i] = fv;
        return 0;
    }

    /* 그 외엔 시퀀스로 해석 */
    PyObject *seq = PySequence_Fast(first, "Tensor expects int size or sequence of floats");
    if (seq == NULL) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    free(self->data);
    self->size = n;
    self->data = alloc_floats(n);
    if (self->data == NULL && n > 0) {
        Py_DECREF(seq);
        PyErr_NoMemory();
        return -1;
    }
    PyObject **items = PySequence_Fast_ITEMS(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        double v = PyFloat_AsDouble(items[i]);
        if (v == -1.0 && PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
        self->data[i] = (float)v;
    }
    Py_DECREF(seq);
    return 0;
}

/* ─── 시퀀스 프로토콜 (__len__/__getitem__/__setitem__) ───────────────── */

static Py_ssize_t Tensor_length(TensorObject *self) {
    return self->size;
}

static PyObject *Tensor_getitem(TensorObject *self, Py_ssize_t i) {
    if (i < 0) i += self->size;
    if (i < 0 || i >= self->size) {
        PyErr_SetString(PyExc_IndexError, "tensor index out of range");
        return NULL;
    }
    return PyFloat_FromDouble((double)self->data[i]);
}

static int Tensor_setitem(TensorObject *self, Py_ssize_t i, PyObject *value) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot delete tensor element");
        return -1;
    }
    if (i < 0) i += self->size;
    if (i < 0 || i >= self->size) {
        PyErr_SetString(PyExc_IndexError, "tensor assignment index out of range");
        return -1;
    }
    double v = PyFloat_AsDouble(value);
    if (v == -1.0 && PyErr_Occurred()) return -1;
    self->data[i] = (float)v;
    return 0;
}

static PySequenceMethods Tensor_as_sequence = {
    .sq_length = (lenfunc)Tensor_length,
    .sq_item = (ssizeargfunc)Tensor_getitem,
    .sq_ass_item = (ssizeobjargproc)Tensor_setitem,
};

/* ─── 산술 (핫루프 GIL 해제 + restrict 로 auto-vectorize 친화) ───────── */

#define DEFINE_VEC_TT(name, op) \
static inline void vec_tt_##name( \
        float * restrict po, \
        const float * restrict pa, \
        const float * restrict pb, \
        Py_ssize_t n) { \
    for (Py_ssize_t i = 0; i < n; ++i) po[i] = pa[i] op pb[i]; \
}

#define DEFINE_VEC_TS(name, op) \
static inline void vec_ts_##name( \
        float * restrict po, \
        const float * restrict pa, \
        float s, \
        Py_ssize_t n) { \
    for (Py_ssize_t i = 0; i < n; ++i) po[i] = pa[i] op s; \
}

#define DEFINE_VEC_ST(name, op) \
static inline void vec_st_##name( \
        float * restrict po, \
        float s, \
        const float * restrict pa, \
        Py_ssize_t n) { \
    for (Py_ssize_t i = 0; i < n; ++i) po[i] = s op pa[i]; \
}

DEFINE_VEC_TT(add, +)
DEFINE_VEC_TT(sub, -)
DEFINE_VEC_TT(mul, *)
DEFINE_VEC_TS(add, +)
DEFINE_VEC_TS(sub, -)
DEFINE_VEC_TS(mul, *)
DEFINE_VEC_ST(add, +)
DEFINE_VEC_ST(sub, -)
DEFINE_VEC_ST(mul, *)

static TensorObject *new_uninit_tensor(Py_ssize_t n) {
    TensorObject *out = (TensorObject *)TensorType.tp_alloc(&TensorType, 0);
    if (out == NULL) return NULL;
    out->size = n;
    out->data = alloc_floats(n);
    if (out->data == NULL && n > 0) {
        Py_DECREF(out);
        PyErr_NoMemory();
        return NULL;
    }
    return out;
}

typedef enum { OP_ADD, OP_SUB, OP_MUL } BinOp;

static PyObject *do_binary(PyObject *a, PyObject *b, BinOp op) {
    int a_is_t = Py_IS_TYPE(a, &TensorType);
    int b_is_t = Py_IS_TYPE(b, &TensorType);

    /* Tensor × Tensor */
    if (a_is_t && b_is_t) {
        TensorObject *ta = (TensorObject *)a;
        TensorObject *tb = (TensorObject *)b;
        if (ta->size != tb->size) {
            PyErr_Format(PyExc_ValueError,
                         "tensor sizes don't match: %zd vs %zd",
                         ta->size, tb->size);
            return NULL;
        }
        TensorObject *out = new_uninit_tensor(ta->size);
        if (out == NULL) return NULL;
        const float *pa = ta->data;
        const float *pb = tb->data;
        float *po = out->data;
        Py_ssize_t n = ta->size;
        Py_BEGIN_ALLOW_THREADS
        switch (op) {
            case OP_ADD: vec_tt_add(po, pa, pb, n); break;
            case OP_SUB: vec_tt_sub(po, pa, pb, n); break;
            case OP_MUL: vec_tt_mul(po, pa, pb, n); break;
        }
        Py_END_ALLOW_THREADS
        return (PyObject *)out;
    }

    /* 스칼라 한쪽 */
    TensorObject *tensor;
    double scalar;
    int scalar_first;
    if (a_is_t) {
        tensor = (TensorObject *)a;
        scalar = PyFloat_AsDouble(b);
        if (scalar == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        scalar_first = 0;
    } else if (b_is_t) {
        tensor = (TensorObject *)b;
        scalar = PyFloat_AsDouble(a);
        if (scalar == -1.0 && PyErr_Occurred()) {
            PyErr_Clear();
            Py_RETURN_NOTIMPLEMENTED;
        }
        scalar_first = 1;
    } else {
        Py_RETURN_NOTIMPLEMENTED;
    }

    TensorObject *out = new_uninit_tensor(tensor->size);
    if (out == NULL) return NULL;
    const float *pa = tensor->data;
    float *po = out->data;
    const float s = (float)scalar;
    Py_ssize_t n = tensor->size;
    Py_BEGIN_ALLOW_THREADS
    if (scalar_first) {
        switch (op) {
            case OP_ADD: vec_st_add(po, s, pa, n); break;
            case OP_SUB: vec_st_sub(po, s, pa, n); break;
            case OP_MUL: vec_st_mul(po, s, pa, n); break;
        }
    } else {
        switch (op) {
            case OP_ADD: vec_ts_add(po, pa, s, n); break;
            case OP_SUB: vec_ts_sub(po, pa, s, n); break;
            case OP_MUL: vec_ts_mul(po, pa, s, n); break;
        }
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

static PyObject *Tensor_nb_add(PyObject *a, PyObject *b) { return do_binary(a, b, OP_ADD); }
static PyObject *Tensor_nb_sub(PyObject *a, PyObject *b) { return do_binary(a, b, OP_SUB); }
static PyObject *Tensor_nb_mul(PyObject *a, PyObject *b) { return do_binary(a, b, OP_MUL); }

static PyNumberMethods Tensor_as_number = {
    .nb_add = Tensor_nb_add,
    .nb_subtract = Tensor_nb_sub,
    .nb_multiply = Tensor_nb_mul,
};

/* ─── 메서드 ───────────────────────────────────────────────────────────── */

static PyObject *Tensor_to_list(TensorObject *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *out = PyList_New(self->size);
    if (out == NULL) return NULL;
    for (Py_ssize_t i = 0; i < self->size; ++i) {
        PyObject *f = PyFloat_FromDouble((double)self->data[i]);
        if (f == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(out, i, f);
    }
    return out;
}

static PyMethodDef Tensor_methods[] = {
    {"to_list", (PyCFunction)Tensor_to_list, METH_NOARGS,
     "이 텐서를 파이썬 list[float] 로 변환해 반환한다."},
    {NULL, NULL, 0, NULL},
};

/* ─── repr ─────────────────────────────────────────────────────────────── */

static PyObject *Tensor_repr(TensorObject *self) {
    Py_ssize_t n = self->size;
    if (n <= 6) {
        PyObject *list = Tensor_to_list(self, NULL);
        if (list == NULL) return NULL;
        PyObject *s = PyUnicode_FromFormat("Tensor(%R)", list);
        Py_DECREF(list);
        return s;
    }
    /* 큰 텐서는 앞 3, 뒤 3 만 미리보기 */
    PyObject *list = Tensor_to_list(self, NULL);
    if (list == NULL) return NULL;
    PyObject *first = PyList_GetSlice(list, 0, 3);
    PyObject *last = PyList_GetSlice(list, n - 3, n);
    Py_DECREF(list);
    if (first == NULL || last == NULL) {
        Py_XDECREF(first);
        Py_XDECREF(last);
        return NULL;
    }
    PyObject *s = PyUnicode_FromFormat(
        "Tensor(size=%zd, %R + ... + %R)", n, first, last);
    Py_DECREF(first);
    Py_DECREF(last);
    return s;
}

/* ─── 타입 ─────────────────────────────────────────────────────────────── */

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pydya._tensor.Tensor",
    .tp_doc = PyDoc_STR("1D float32 contiguous tensor (C 레벨 primitive)."),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Tensor_init,
    .tp_dealloc = (destructor)Tensor_dealloc,
    .tp_repr = (reprfunc)Tensor_repr,
    .tp_as_sequence = &Tensor_as_sequence,
    .tp_as_number = &Tensor_as_number,
    .tp_methods = Tensor_methods,
};

/* ─── 모듈 ─────────────────────────────────────────────────────────────── */

static struct PyModuleDef _tensormodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pydya._tensor",
    .m_doc = "C 레벨 1D float32 Tensor primitive.",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit__tensor(void) {
    if (PyType_Ready(&TensorType) < 0) return NULL;
    PyObject *m = PyModule_Create(&_tensormodule);
    if (m == NULL) return NULL;
    Py_INCREF(&TensorType);
    if (PyModule_AddObject(m, "Tensor", (PyObject *)&TensorType) < 0) {
        Py_DECREF(&TensorType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
