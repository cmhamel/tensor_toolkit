#pragma once

namespace ttk {

// ---------------- helpers ----------------
constexpr int sym_length(int dim) { 
    return dim * (dim + 1) / 2; 
}
constexpr int full_length(int dim) { 
    return dim * dim; 
}

// // ---------------- recursive length calculator ----------------
// template<int D, bool... Syms>
// struct _TensorLength;

// // base case: no symmetry flags left
// template<int D>
// struct _TensorLength<D> {
//     static constexpr int value = D;
// };

// // recursive case: fold over symmetry flags
// template<int D, bool Head, bool... Tail>
// struct _TensorLength<D, Head, Tail...> {
// private:
//     // multiply by next dimension considering symmetry
//     static constexpr int next = Head ? sym_length(D) : full_length(D);

// public:
//     // fold recursively
//     static constexpr int value = next * _TensorLength<D, Tail...>::value / D;
// };

template<int D, int O, bool... Syms>
struct _TensorLength;

// 1st order
template<int D>
struct _TensorLength<D, 1> {
    static constexpr int value = D;
};

// 2nd order
template<int D>
struct _TensorLength<D, 2, false> {
    static constexpr int value = D * D;
};

template<int D>
struct _TensorLength<D, 2, true> {
    static constexpr int value = D * (D + 1) / 2;
};

// 3rd order
template<int D>
struct _TensorLength<D, 3, false, false> {
    static constexpr int value = D * D * D;
};

// 4th order
template<int D>
struct _TensorLength<D, 4, false, false, false> {
    static constexpr int value = D * D * D * D;
};

template<int D>
struct _TensorLength<D, 4, false, true, false> {
    static constexpr int value = D * D * (D + 1) * (D + 1) / 2 / 2;
};

// ---------------- public trait ----------------
template<int D, int O, bool... Syms>
struct TensorLength {
    // TODO add more compile time checks that e.g.
    // order/dimension or sensible/supported
    // sanity check: number of symmetry flags must be order-1
    static_assert(
        sizeof...(Syms) == (O > 0 ? O - 1 : 0),
        "Number of symmetry flags must be O-1"
    );
    // static constexpr int value =
    //     (O == 0) ? 1 :               // scalar
    //     (O == 1) ? D :               // vector
    //     _TensorLength<D, Syms...>::value;
    static constexpr int value = 
        (O == 0) ? 1 :
        _TensorLength<D, O, Syms...>::value;
};

} // end namespace ttk
