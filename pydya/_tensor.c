/*
 * pydya._tensor — C 레벨 N-D float32 Tensor primitive.
 *
 * Nadya 의 tensor 가 컴파일러 단의 primitive 데이터 타입이듯, Pydya 도
 * 텐서 연산을 C 레벨에서 수행해 (1) 자동 벡터화(-O3 -march=native 로
 * 컴파일러가 SIMD 깔아 줌)와 (2) 산술 핫루프 GIL 해제를 통한 진짜 멀티코어
 * 잠재력을 확보한다.
 *
 * 메모리 레이아웃: float32 contiguous row-major. shape[ndim] 으로 형상 보유,
 * stride 는 shape 로부터 계산(저장 안 함, contiguous 가정).
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
    Py_ssize_t size;     /* 전체 원소 수 (shape 곱) */
    Py_ssize_t ndim;     /* 차원 수 */
    Py_ssize_t *shape;   /* shape[ndim] (heap) */
} TensorObject;

static PyTypeObject TensorType;  /* forward */

static void Tensor_dealloc(TensorObject *self) {
    free(self->data);
    free(self->shape);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

/* ─── 도우미: shape/구조 할당 ──────────────────────────────────────────── */

static int tensor_set_shape(TensorObject *self, const Py_ssize_t *dims, Py_ssize_t ndim) {
    free(self->shape);
    self->shape = NULL;
    self->ndim = ndim;
    Py_ssize_t total = 1;
    if (ndim > 0) {
        self->shape = (Py_ssize_t *)malloc((size_t)ndim * sizeof(Py_ssize_t));
        if (self->shape == NULL) {
            PyErr_NoMemory();
            return -1;
        }
        for (Py_ssize_t i = 0; i < ndim; ++i) {
            if (dims[i] < 0) {
                PyErr_SetString(PyExc_ValueError, "shape dimensions must be non-negative");
                return -1;
            }
            self->shape[i] = dims[i];
            total *= dims[i];
        }
    }
    self->size = total;
    return 0;
}

static int tensor_allocate_data(TensorObject *self) {
    free(self->data);
    self->data = alloc_floats(self->size);
    if (self->data == NULL && self->size > 0) {
        PyErr_NoMemory();
        return -1;
    }
    return 0;
}

static void tensor_fill(TensorObject *self, float value) {
    for (Py_ssize_t i = 0; i < self->size; ++i) self->data[i] = value;
}

/* shape 튜플을 ssize 배열로. 성공 시 dims 채우고 ndim 반환, 실패 시 -1. */
static Py_ssize_t parse_shape_tuple(PyObject *tup, Py_ssize_t **dims_out) {
    Py_ssize_t ndim = PyTuple_GET_SIZE(tup);
    if (ndim == 0) {
        PyErr_SetString(PyExc_ValueError, "shape tuple must not be empty");
        return -1;
    }
    Py_ssize_t *dims = (Py_ssize_t *)malloc((size_t)ndim * sizeof(Py_ssize_t));
    if (dims == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < ndim; ++i) {
        PyObject *item = PyTuple_GET_ITEM(tup, i);
        if (!PyLong_Check(item)) {
            free(dims);
            PyErr_SetString(PyExc_TypeError, "shape tuple must contain ints");
            return -1;
        }
        Py_ssize_t v = PyLong_AsSsize_t(item);
        if (v == -1 && PyErr_Occurred()) {
            free(dims);
            return -1;
        }
        if (v < 0) {
            free(dims);
            PyErr_SetString(PyExc_ValueError, "shape dimensions must be non-negative");
            return -1;
        }
        dims[i] = v;
    }
    *dims_out = dims;
    return ndim;
}

/* 중첩 리스트의 깊이/형상 추론 (모든 가지의 길이가 같다고 가정). */
static int infer_nested_shape(PyObject *seq, Py_ssize_t **dims_out, Py_ssize_t *ndim_out) {
    Py_ssize_t cap = 4;
    Py_ssize_t ndim = 0;
    Py_ssize_t *dims = (Py_ssize_t *)malloc((size_t)cap * sizeof(Py_ssize_t));
    if (dims == NULL) {
        PyErr_NoMemory();
        return -1;
    }
    PyObject *cur = seq;
    Py_INCREF(cur);
    while (PyList_Check(cur) || PyTuple_Check(cur)) {
        Py_ssize_t len = PySequence_Size(cur);
        if (len < 0) { Py_DECREF(cur); free(dims); return -1; }
        if (ndim >= cap) {
            cap *= 2;
            Py_ssize_t *nd = (Py_ssize_t *)realloc(dims, (size_t)cap * sizeof(Py_ssize_t));
            if (nd == NULL) { Py_DECREF(cur); free(dims); PyErr_NoMemory(); return -1; }
            dims = nd;
        }
        dims[ndim++] = len;
        if (len == 0) break;
        PyObject *first = PySequence_GetItem(cur, 0);
        Py_DECREF(cur);
        if (first == NULL) { free(dims); return -1; }
        cur = first;
    }
    Py_DECREF(cur);
    *dims_out = dims;
    *ndim_out = ndim;
    return 0;
}

/* 중첩 시퀀스를 contiguous data 로 평탄화. ndim/shape 가 이미 결정돼야 함. */
static int flatten_nested(PyObject *seq, float *out, const Py_ssize_t *shape, Py_ssize_t ndim) {
    if (ndim == 1) {
        PyObject *fast = PySequence_Fast(seq, "expected sequence");
        if (fast == NULL) return -1;
        Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
        if (n != shape[0]) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_ValueError, "ragged nested sequence");
            return -1;
        }
        PyObject **items = PySequence_Fast_ITEMS(fast);
        for (Py_ssize_t i = 0; i < n; ++i) {
            double v = PyFloat_AsDouble(items[i]);
            if (v == -1.0 && PyErr_Occurred()) { Py_DECREF(fast); return -1; }
            out[i] = (float)v;
        }
        Py_DECREF(fast);
        return 0;
    }
    PyObject *fast = PySequence_Fast(seq, "expected sequence");
    if (fast == NULL) return -1;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n != shape[0]) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_ValueError, "ragged nested sequence");
        return -1;
    }
    Py_ssize_t inner = 1;
    for (Py_ssize_t k = 1; k < ndim; ++k) inner *= shape[k];
    PyObject **items = PySequence_Fast_ITEMS(fast);
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (flatten_nested(items[i], out + i * inner, shape + 1, ndim - 1) < 0) {
            Py_DECREF(fast);
            return -1;
        }
    }
    Py_DECREF(fast);
    return 0;
}

static int Tensor_init(TensorObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"data", "fill", NULL};
    PyObject *first = NULL;
    double fill = 0.0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|d", kwlist, &first, &fill)) {
        return -1;
    }

    /* 정수면 1D size 로 해석해 fill 로 초기화 (back-compat) */
    if (PyLong_CheckExact(first)) {
        Py_ssize_t n = PyLong_AsSsize_t(first);
        if (n == -1 && PyErr_Occurred()) return -1;
        if (n < 0) {
            PyErr_SetString(PyExc_ValueError, "size must be non-negative");
            return -1;
        }
        Py_ssize_t dims[1] = {n};
        if (tensor_set_shape(self, dims, 1) < 0) return -1;
        if (tensor_allocate_data(self) < 0) return -1;
        tensor_fill(self, (float)fill);
        return 0;
    }

    /* 튜플인데 원소가 모두 int 면 shape 로 해석 (numpy 식: zeros((2,3))).
     * 그 외 (float 섞임, 빈 튜플) 는 아래 시퀀스 경로로 떨어진다. */
    if (PyTuple_Check(first)) {
        Py_ssize_t klen = PyTuple_GET_SIZE(first);
        int all_int = klen > 0;
        for (Py_ssize_t i = 0; i < klen; ++i) {
            if (!PyLong_Check(PyTuple_GET_ITEM(first, i))) { all_int = 0; break; }
        }
        if (all_int) {
            Py_ssize_t *dims = NULL;
            Py_ssize_t ndim = parse_shape_tuple(first, &dims);
            if (ndim < 0) return -1;
            int rc = tensor_set_shape(self, dims, ndim);
            free(dims);
            if (rc < 0) return -1;
            if (tensor_allocate_data(self) < 0) return -1;
            tensor_fill(self, (float)fill);
            return 0;
        }
        /* float 섞인 튜플은 시퀀스로 폴백 */
    }

    /* 리스트/시퀀스: 중첩 깊이로 ndim 추론, 그 형상으로 평탄화 */
    if (PyList_Check(first) || PySequence_Check(first)) {
        Py_ssize_t *dims = NULL;
        Py_ssize_t ndim = 0;
        if (infer_nested_shape(first, &dims, &ndim) < 0) return -1;
        int rc = tensor_set_shape(self, dims, ndim);
        free(dims);
        if (rc < 0) return -1;
        if (tensor_allocate_data(self) < 0) return -1;
        if (self->size > 0 && flatten_nested(first, self->data, self->shape, self->ndim) < 0) {
            return -1;
        }
        return 0;
    }

    PyErr_SetString(PyExc_TypeError,
                    "Tensor expects int size, shape tuple, or (nested) sequence of floats");
    return -1;
}

/* ─── 시퀀스 프로토콜 ──────────────────────────────────────────────────── */

static Py_ssize_t Tensor_length(TensorObject *self) {
    /* 1D 는 원소 수, N-D 는 첫 번째 차원 길이 (numpy 와 동일 의미) */
    if (self->ndim == 0) return 0;
    return self->shape[0];
}

/* 튜플 (i0, i1, ..., i_{ndim-1}) → row-major flat offset 변환. */
static int tuple_to_offset(TensorObject *self, PyObject *key, Py_ssize_t *out_offset) {
    Py_ssize_t klen = PyTuple_GET_SIZE(key);
    if (klen != self->ndim) {
        PyErr_Format(PyExc_IndexError,
                     "tuple index has %zd element(s) but tensor is %zdD",
                     klen, self->ndim);
        return -1;
    }
    Py_ssize_t accum = 0;
    Py_ssize_t mult = 1;
    for (Py_ssize_t d = self->ndim - 1; d >= 0; --d) {
        PyObject *idx_obj = PyTuple_GET_ITEM(key, d);
        if (!PyLong_Check(idx_obj)) {
            PyErr_SetString(PyExc_TypeError, "tuple index elements must be int");
            return -1;
        }
        Py_ssize_t idx = PyLong_AsSsize_t(idx_obj);
        if (idx == -1 && PyErr_Occurred()) return -1;
        if (idx < 0) idx += self->shape[d];
        if (idx < 0 || idx >= self->shape[d]) {
            PyErr_Format(PyExc_IndexError,
                         "index %zd out of range for dim %zd (size %zd)",
                         idx, d, self->shape[d]);
            return -1;
        }
        accum += idx * mult;
        mult *= self->shape[d];
    }
    *out_offset = accum;
    return 0;
}

static PyObject *Tensor_subscript(TensorObject *self, PyObject *key) {
    if (PyLong_Check(key)) {
        if (self->ndim != 1) {
            PyErr_Format(PyExc_TypeError,
                "single int index requires 1D tensor (got %zdD); use a tuple",
                self->ndim);
            return NULL;
        }
        Py_ssize_t i = PyLong_AsSsize_t(key);
        if (i == -1 && PyErr_Occurred()) return NULL;
        if (i < 0) i += self->size;
        if (i < 0 || i >= self->size) {
            PyErr_SetString(PyExc_IndexError, "tensor index out of range");
            return NULL;
        }
        return PyFloat_FromDouble((double)self->data[i]);
    }
    if (PyTuple_Check(key)) {
        Py_ssize_t off;
        if (tuple_to_offset(self, key, &off) < 0) return NULL;
        return PyFloat_FromDouble((double)self->data[off]);
    }
    PyErr_SetString(PyExc_TypeError, "tensor indices must be int (1D) or tuple of ints");
    return NULL;
}

static int Tensor_ass_subscript(TensorObject *self, PyObject *key, PyObject *value) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "cannot delete tensor element");
        return -1;
    }
    Py_ssize_t off;
    if (PyLong_Check(key)) {
        if (self->ndim != 1) {
            PyErr_Format(PyExc_TypeError,
                "single int index requires 1D tensor (got %zdD); use a tuple",
                self->ndim);
            return -1;
        }
        Py_ssize_t i = PyLong_AsSsize_t(key);
        if (i == -1 && PyErr_Occurred()) return -1;
        if (i < 0) i += self->size;
        if (i < 0 || i >= self->size) {
            PyErr_SetString(PyExc_IndexError, "tensor assignment index out of range");
            return -1;
        }
        off = i;
    } else if (PyTuple_Check(key)) {
        if (tuple_to_offset(self, key, &off) < 0) return -1;
    } else {
        PyErr_SetString(PyExc_TypeError, "tensor indices must be int (1D) or tuple of ints");
        return -1;
    }
    double v = PyFloat_AsDouble(value);
    if (v == -1.0 && PyErr_Occurred()) return -1;
    self->data[off] = (float)v;
    return 0;
}

static PySequenceMethods Tensor_as_sequence = {
    .sq_length = (lenfunc)Tensor_length,
};

static PyMappingMethods Tensor_as_mapping = {
    .mp_length = (lenfunc)Tensor_length,
    .mp_subscript = (binaryfunc)Tensor_subscript,
    .mp_ass_subscript = (objobjargproc)Tensor_ass_subscript,
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

static int shapes_equal(TensorObject *a, TensorObject *b) {
    if (a->ndim != b->ndim) return 0;
    for (Py_ssize_t i = 0; i < a->ndim; ++i) {
        if (a->shape[i] != b->shape[i]) return 0;
    }
    return 1;
}

static PyObject *shape_to_tuple(const Py_ssize_t *shape, Py_ssize_t ndim) {
    PyObject *tup = PyTuple_New(ndim);
    if (tup == NULL) return NULL;
    for (Py_ssize_t i = 0; i < ndim; ++i) {
        PyObject *n = PyLong_FromSsize_t(shape[i]);
        if (n == NULL) { Py_DECREF(tup); return NULL; }
        PyTuple_SET_ITEM(tup, i, n);
    }
    return tup;
}

static TensorObject *new_uninit_tensor_with_shape(const Py_ssize_t *dims, Py_ssize_t ndim) {
    TensorObject *out = (TensorObject *)TensorType.tp_alloc(&TensorType, 0);
    if (out == NULL) return NULL;
    if (tensor_set_shape(out, dims, ndim) < 0) {
        Py_DECREF(out);
        return NULL;
    }
    if (tensor_allocate_data(out) < 0) {
        Py_DECREF(out);
        return NULL;
    }
    return out;
}

static TensorObject *new_uninit_tensor_like(TensorObject *src) {
    return new_uninit_tensor_with_shape(src->shape, src->ndim);
}

typedef enum { OP_ADD, OP_SUB, OP_MUL } BinOp;

static PyObject *do_binary(PyObject *a, PyObject *b, BinOp op) {
    int a_is_t = Py_IS_TYPE(a, &TensorType);
    int b_is_t = Py_IS_TYPE(b, &TensorType);

    /* Tensor × Tensor (같은 shape 필수, 일반 broadcasting 미구현) */
    if (a_is_t && b_is_t) {
        TensorObject *ta = (TensorObject *)a;
        TensorObject *tb = (TensorObject *)b;
        if (!shapes_equal(ta, tb)) {
            PyObject *sa = shape_to_tuple(ta->shape, ta->ndim);
            PyObject *sb = shape_to_tuple(tb->shape, tb->ndim);
            PyErr_Format(PyExc_ValueError,
                         "tensor shapes don't match: %R vs %R", sa, sb);
            Py_XDECREF(sa);
            Py_XDECREF(sb);
            return NULL;
        }
        TensorObject *out = new_uninit_tensor_like(ta);
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

    /* 스칼라 한쪽 (shape 보존) */
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

    TensorObject *out = new_uninit_tensor_like(tensor);
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

/* @ operator → matmul (forward declaration of tensor_matmul 가 아래에 있음) */
static PyObject *tensor_matmul(PyObject *self, PyObject *args);
static PyObject *Tensor_nb_matmul(PyObject *a, PyObject *b) {
    PyObject *args = PyTuple_Pack(2, a, b);
    if (args == NULL) return NULL;
    PyObject *result = tensor_matmul(NULL, args);
    Py_DECREF(args);
    return result;
}

static PyNumberMethods Tensor_as_number = {
    .nb_add = Tensor_nb_add,
    .nb_subtract = Tensor_nb_sub,
    .nb_multiply = Tensor_nb_mul,
    .nb_matrix_multiply = Tensor_nb_matmul,
};

/* ─── 메서드 ───────────────────────────────────────────────────────────── */

static PyObject *to_nested_list(const float *data, const Py_ssize_t *shape, Py_ssize_t ndim) {
    if (ndim == 0) {
        return PyFloat_FromDouble((double)data[0]);
    }
    Py_ssize_t outer = shape[0];
    PyObject *out = PyList_New(outer);
    if (out == NULL) return NULL;
    if (ndim == 1) {
        for (Py_ssize_t i = 0; i < outer; ++i) {
            PyObject *f = PyFloat_FromDouble((double)data[i]);
            if (f == NULL) { Py_DECREF(out); return NULL; }
            PyList_SET_ITEM(out, i, f);
        }
        return out;
    }
    Py_ssize_t inner = 1;
    for (Py_ssize_t k = 1; k < ndim; ++k) inner *= shape[k];
    for (Py_ssize_t i = 0; i < outer; ++i) {
        PyObject *sub = to_nested_list(data + i * inner, shape + 1, ndim - 1);
        if (sub == NULL) { Py_DECREF(out); return NULL; }
        PyList_SET_ITEM(out, i, sub);
    }
    return out;
}

static PyObject *Tensor_to_list(TensorObject *self, PyObject *Py_UNUSED(ignored)) {
    return to_nested_list(self->data, self->shape, self->ndim);
}

static PyObject *Tensor_get_shape(TensorObject *self, void *Py_UNUSED(closure)) {
    return shape_to_tuple(self->shape, self->ndim);
}

static PyObject *Tensor_get_ndim(TensorObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromSsize_t(self->ndim);
}

static PyObject *Tensor_get_size(TensorObject *self, void *Py_UNUSED(closure)) {
    return PyLong_FromSsize_t(self->size);
}

static PyGetSetDef Tensor_getset[] = {
    {"shape", (getter)Tensor_get_shape, NULL, "텐서 형상 튜플.", NULL},
    {"ndim", (getter)Tensor_get_ndim, NULL, "차원 수.", NULL},
    {"size", (getter)Tensor_get_size, NULL, "전체 원소 수.", NULL},
    {NULL, NULL, NULL, NULL, NULL},
};

static PyMethodDef Tensor_methods[] = {
    {"to_list", (PyCFunction)Tensor_to_list, METH_NOARGS,
     "텐서를 파이썬 list 로 변환해 반환한다 (N-D 면 중첩 리스트)."},
    {NULL, NULL, 0, NULL},
};

/* ─── 모듈 레벨 융합 커널 (Phase 3) ────────────────────────────────────── */

/* a * b + c — 세 텐서 모두 같은 size, 단일 할당 단일 메모리 순회.
 * 미융합 a*b+c 는 임시 텐서 2개를 만들고 메모리를 두 번 더 순회한다. */
static PyObject *tensor_madd(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *a_obj, *b_obj, *c_obj;
    if (!PyArg_ParseTuple(args, "OOO", &a_obj, &b_obj, &c_obj)) return NULL;
    if (!Py_IS_TYPE(a_obj, &TensorType) ||
        !Py_IS_TYPE(b_obj, &TensorType) ||
        !Py_IS_TYPE(c_obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "madd requires three Tensor arguments");
        return NULL;
    }
    TensorObject *ta = (TensorObject *)a_obj;
    TensorObject *tb = (TensorObject *)b_obj;
    TensorObject *tc = (TensorObject *)c_obj;
    if (!shapes_equal(ta, tb) || !shapes_equal(ta, tc)) {
        PyErr_SetString(PyExc_ValueError, "madd: tensor shapes must match");
        return NULL;
    }
    TensorObject *out = new_uninit_tensor_like(ta);
    if (out == NULL) return NULL;
    const float * restrict pa = ta->data;
    const float * restrict pb = tb->data;
    const float * restrict pc = tc->data;
    float * restrict po = out->data;
    Py_ssize_t n = ta->size;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        po[i] = pa[i] * pb[i] + pc[i];
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* matmul: 2D W (rows, cols) × 1D x (cols,) → 1D out (rows,).
 * inner loop 는 직선 코드 — 컴파일러 auto-vectorizer 가 SIMD 깐다. */
static PyObject *tensor_matmul(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *W_obj, *x_obj;
    if (!PyArg_ParseTuple(args, "OO", &W_obj, &x_obj)) return NULL;
    if (!Py_IS_TYPE(W_obj, &TensorType) || !Py_IS_TYPE(x_obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "matmul requires two Tensor arguments");
        return NULL;
    }
    TensorObject *W = (TensorObject *)W_obj;
    TensorObject *x = (TensorObject *)x_obj;
    if (W->ndim != 2 || x->ndim != 1) {
        PyErr_Format(PyExc_ValueError,
                     "matmul currently supports 2D @ 1D only (got %zdD @ %zdD)",
                     W->ndim, x->ndim);
        return NULL;
    }
    Py_ssize_t rows = W->shape[0];
    Py_ssize_t cols = W->shape[1];
    if (cols != x->shape[0]) {
        PyErr_Format(PyExc_ValueError,
                     "matmul shape mismatch: W is (%zd, %zd), x is (%zd,)",
                     rows, cols, x->shape[0]);
        return NULL;
    }
    Py_ssize_t out_dims[1] = {rows};
    TensorObject *out = new_uninit_tensor_with_shape(out_dims, 1);
    if (out == NULL) return NULL;
    const float * restrict pw = W->data;
    const float * restrict px = x->data;
    float * restrict po = out->data;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < rows; ++i) {
        const float * restrict row = pw + i * cols;
        float acc = 0.0f;
        for (Py_ssize_t j = 0; j < cols; ++j) {
            acc += row[j] * px[j];
        }
        po[i] = acc;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* relu: element-wise max(0, x). */
static PyObject *tensor_relu(PyObject *Py_UNUSED(self), PyObject *arg) {
    if (!Py_IS_TYPE(arg, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "relu requires a Tensor argument");
        return NULL;
    }
    TensorObject *t = (TensorObject *)arg;
    TensorObject *out = new_uninit_tensor_like(t);
    if (out == NULL) return NULL;
    const float * restrict pi = t->data;
    float * restrict po = out->data;
    Py_ssize_t n = t->size;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        float v = pi[i];
        po[i] = v > 0.0f ? v : 0.0f;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* linear_relu: relu(W @ x + b) — 단일 패스 융합 커널 (Phase 3 의 텐서 버전,
 * Optimium 탐구3 의 Conv+ReLU 융합 대응). 임시 텐서 2개 할당과 메모리
 * 두 번 추가 순회를 제거한다. */
static PyObject *tensor_linear_relu(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *W_obj, *x_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OOO", &W_obj, &x_obj, &b_obj)) return NULL;
    if (!Py_IS_TYPE(W_obj, &TensorType) ||
        !Py_IS_TYPE(x_obj, &TensorType) ||
        !Py_IS_TYPE(b_obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "linear_relu requires three Tensor arguments");
        return NULL;
    }
    TensorObject *W = (TensorObject *)W_obj;
    TensorObject *x = (TensorObject *)x_obj;
    TensorObject *b = (TensorObject *)b_obj;
    if (W->ndim != 2 || x->ndim != 1 || b->ndim != 1) {
        PyErr_SetString(PyExc_ValueError,
                        "linear_relu: W must be 2D, x and b must be 1D");
        return NULL;
    }
    Py_ssize_t rows = W->shape[0];
    Py_ssize_t cols = W->shape[1];
    if (cols != x->shape[0] || rows != b->shape[0]) {
        PyErr_Format(PyExc_ValueError,
                     "linear_relu shape mismatch: W (%zd, %zd), x (%zd,), b (%zd,)",
                     rows, cols, x->shape[0], b->shape[0]);
        return NULL;
    }
    Py_ssize_t out_dims[1] = {rows};
    TensorObject *out = new_uninit_tensor_with_shape(out_dims, 1);
    if (out == NULL) return NULL;
    const float * restrict pw = W->data;
    const float * restrict px = x->data;
    const float * restrict pb = b->data;
    float * restrict po = out->data;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < rows; ++i) {
        const float * restrict row = pw + i * cols;
        float acc = pb[i];
        for (Py_ssize_t j = 0; j < cols; ++j) {
            acc += row[j] * px[j];
        }
        po[i] = acc > 0.0f ? acc : 0.0f;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

/* ─── Scalar 변종 (auto-vectorize 끄기) — 벤치마크 비교용 ──────────────────
 *
 * 같은 알고리즘을 *컴파일러 SIMD 패스를 끈* 형태로 따로 빌드한다. C 레벨
 * 실행 비용 (Python 인터프리터 대비 raw C 핫루프) 만 측정하기 위해.
 *
 *   stage A → B → C → D 로 비교:
 *     A: Pure Python
 *     B: C scalar    (이 섹션)  ── A→B 가 'C-Level Tensor' 기여
 *     C: C vectorized (위 일반 ops)  ── B→C 가 'Vector 최적화' 기여
 *     D: C vec + fused linear_relu   ── C→D 가 '표현식 융합' 기여
 *
 * pragma 로 함수 단위 -fno-tree-vectorize 적용. restrict 도 일부러 빼서
 * 컴파일러가 aliasing 가정해 SIMD 못 쓰게 한다.
 */
#pragma GCC push_options
#pragma GCC optimize("no-tree-vectorize", "no-tree-slp-vectorize")

static PyObject *tensor_matmul_scalar(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *W_obj, *x_obj;
    if (!PyArg_ParseTuple(args, "OO", &W_obj, &x_obj)) return NULL;
    if (!Py_IS_TYPE(W_obj, &TensorType) || !Py_IS_TYPE(x_obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "matmul_scalar requires two Tensor arguments");
        return NULL;
    }
    TensorObject *W = (TensorObject *)W_obj;
    TensorObject *x = (TensorObject *)x_obj;
    if (W->ndim != 2 || x->ndim != 1) {
        PyErr_SetString(PyExc_ValueError, "matmul_scalar supports 2D @ 1D only");
        return NULL;
    }
    Py_ssize_t rows = W->shape[0];
    Py_ssize_t cols = W->shape[1];
    if (cols != x->shape[0]) {
        PyErr_SetString(PyExc_ValueError, "matmul_scalar shape mismatch");
        return NULL;
    }
    Py_ssize_t out_dims[1] = {rows};
    TensorObject *out = new_uninit_tensor_with_shape(out_dims, 1);
    if (out == NULL) return NULL;
    const float *pw = W->data;
    const float *px = x->data;
    float *po = out->data;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < rows; ++i) {
        const float *row = pw + i * cols;
        float acc = 0.0f;
        for (Py_ssize_t j = 0; j < cols; ++j) {
            acc += row[j] * px[j];
        }
        po[i] = acc;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

static PyObject *tensor_add_scalar(PyObject *Py_UNUSED(self), PyObject *args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) return NULL;
    if (!Py_IS_TYPE(a_obj, &TensorType) || !Py_IS_TYPE(b_obj, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "add_scalar requires two Tensors");
        return NULL;
    }
    TensorObject *a = (TensorObject *)a_obj;
    TensorObject *b = (TensorObject *)b_obj;
    if (!shapes_equal(a, b)) {
        PyErr_SetString(PyExc_ValueError, "add_scalar shape mismatch");
        return NULL;
    }
    TensorObject *out = new_uninit_tensor_like(a);
    if (out == NULL) return NULL;
    const float *pa = a->data;
    const float *pb = b->data;
    float *po = out->data;
    Py_ssize_t n = a->size;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) po[i] = pa[i] + pb[i];
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

static PyObject *tensor_relu_scalar(PyObject *Py_UNUSED(self), PyObject *arg) {
    if (!Py_IS_TYPE(arg, &TensorType)) {
        PyErr_SetString(PyExc_TypeError, "relu_scalar requires a Tensor");
        return NULL;
    }
    TensorObject *t = (TensorObject *)arg;
    TensorObject *out = new_uninit_tensor_like(t);
    if (out == NULL) return NULL;
    const float *pi = t->data;
    float *po = out->data;
    Py_ssize_t n = t->size;
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        float v = pi[i];
        po[i] = v > 0.0f ? v : 0.0f;
    }
    Py_END_ALLOW_THREADS
    return (PyObject *)out;
}

#pragma GCC pop_options

static PyMethodDef _tensor_module_methods[] = {
    {"madd", tensor_madd, METH_VARARGS,
     "a * b + c element-wise (fused). 세 텐서 모두 같은 shape 여야 한다."},
    {"matmul", tensor_matmul, METH_VARARGS,
     "2D W (rows, cols) × 1D x (cols,) → 1D out (rows,)."},
    {"relu", tensor_relu, METH_O,
     "element-wise max(0, x). 입력과 동일 shape 텐서 반환."},
    {"linear_relu", tensor_linear_relu, METH_VARARGS,
     "relu(W @ x + b) 단일 패스 융합. W 2D, x/b 1D, 같은 행 수."},
    {"matmul_scalar", tensor_matmul_scalar, METH_VARARGS,
     "[벤치마크용] 같은 matmul 을 컴파일러 auto-vectorize 끈 채로 실행."},
    {"relu_scalar", tensor_relu_scalar, METH_O,
     "[벤치마크용] auto-vectorize 끈 relu."},
    {"add_scalar", tensor_add_scalar, METH_VARARGS,
     "[벤치마크용] auto-vectorize 끈 element-wise add."},
    {NULL, NULL, 0, NULL},
};

/* ─── repr ─────────────────────────────────────────────────────────────── */

static PyObject *Tensor_repr(TensorObject *self) {
    /* 1D 작은 텐서는 list 그대로, 큰 1D 는 앞/뒤 미리보기,
     * N-D 는 shape 와 중첩 리스트(혹은 평탄 앞/뒤 미리보기). */
    if (self->ndim == 1) {
        Py_ssize_t n = self->size;
        if (n <= 6) {
            PyObject *list = Tensor_to_list(self, NULL);
            if (list == NULL) return NULL;
            PyObject *s = PyUnicode_FromFormat("Tensor(%R)", list);
            Py_DECREF(list);
            return s;
        }
        PyObject *list = Tensor_to_list(self, NULL);
        if (list == NULL) return NULL;
        PyObject *first = PyList_GetSlice(list, 0, 3);
        PyObject *last = PyList_GetSlice(list, n - 3, n);
        Py_DECREF(list);
        if (first == NULL || last == NULL) {
            Py_XDECREF(first); Py_XDECREF(last);
            return NULL;
        }
        PyObject *s = PyUnicode_FromFormat(
            "Tensor(size=%zd, %R + ... + %R)", n, first, last);
        Py_DECREF(first); Py_DECREF(last);
        return s;
    }
    /* N-D: 작으면 nested list, 크면 shape 만 */
    PyObject *shape = shape_to_tuple(self->shape, self->ndim);
    if (shape == NULL) return NULL;
    if (self->size <= 24) {
        PyObject *nested = Tensor_to_list(self, NULL);
        if (nested == NULL) { Py_DECREF(shape); return NULL; }
        PyObject *s = PyUnicode_FromFormat("Tensor(shape=%R, %R)", shape, nested);
        Py_DECREF(shape); Py_DECREF(nested);
        return s;
    }
    PyObject *s = PyUnicode_FromFormat(
        "Tensor(shape=%R, size=%zd)", shape, self->size);
    Py_DECREF(shape);
    return s;
}

/* ─── 타입 ─────────────────────────────────────────────────────────────── */

static PyTypeObject TensorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pydya._tensor.Tensor",
    .tp_doc = PyDoc_STR("N-D float32 contiguous tensor (C 레벨 primitive)."),
    .tp_basicsize = sizeof(TensorObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)Tensor_init,
    .tp_dealloc = (destructor)Tensor_dealloc,
    .tp_repr = (reprfunc)Tensor_repr,
    .tp_as_sequence = &Tensor_as_sequence,
    .tp_as_mapping = &Tensor_as_mapping,
    .tp_as_number = &Tensor_as_number,
    .tp_methods = Tensor_methods,
    .tp_getset = Tensor_getset,
};

/* ─── 모듈 ─────────────────────────────────────────────────────────────── */

static struct PyModuleDef _tensormodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pydya._tensor",
    .m_doc = "C 레벨 1D float32 Tensor primitive.",
    .m_size = -1,
    .m_methods = _tensor_module_methods,
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
