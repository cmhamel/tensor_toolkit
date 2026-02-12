#pragma once
#include <ttk/aliases.hpp>
#include "unary_operators.hpp"

namespace ttk {

template<Scalar T>
T __cos_of_acos_divided_by_3(const T& x) {
    
    const T x2 = x * x;
    const T x4 = x2 * x2;

    const T numer = 0.866025403784438713 + 2.12714890259493060 * x + ( 
                (1.89202064815951569 + 0.739603278343401613 * x) * x2 +
                (0.121973926953064794 + x * (
                    0.00655637626263929360 + 0.0000390884982780803443 * x) 
                ) * x4 
            );
    const T denom = 1.0 + 2.26376989330935617 * x + (
                (1.80461009751278976 + 0.603976798217196003 * x) * x2 +
                (0.0783255761115461708 + 0.00268525944538021629 * x) * x4 
            );
    return numer / denom;
}

template<Scalar T, int M>
std::pair<Vector<T, M, 3>, Tensor2<T, M, 3>> eigen_non_nomalized(const SymmetricTensor2<T, M, 3>& A) {

    T cxx = A(0, 0), cyy = A(1, 1), czz = A(2, 2);
    const T cxy = A(0, 1), cyz = A(1, 2), czx = A(2, 0);
    const T c1 = (cxx + cyy + czz) / (3.0);

    cxx -= c1, cyy -= c1, czz -= c1;
    const T cxy_cxy = cxy * cxy;
    const T cyz_cyz = cyz * cyz;
    const T czx_czx = czx * czx;
    const T cxx_cyy = cxx * cyy;
    const T c2 = cxx_cyy + cyy * czz + czz * cxx - 
                 cxy_cxy - cyz_cyz - czx_czx;
    const bool c2Negative = (c2 < 0.0);
    const T denom = __ifelse(c2Negative, c2, 1.0);
    const T threeOverA = __ifelse(c2Negative, -3.0 / denom, T(1.0));
    const T sqrtThreeOverA = __ifelse(c2Negative, sqrt(threeOverA), 1.0);

    const T c3 = cxx * cyz_cyz + cyy * czx_czx - 
                 2.0 * cxy * cyz * czx + 
                 czz * (cxy_cxy - cxx_cyy);

    const T rr = -0.5 * c3 * threeOverA * sqrtThreeOverA;
    
    const T arg = min(abs(rr), 1.0); // Check in the case rr = -1-eps
    
    const T cos_thd3 = __cos_of_acos_divided_by_3(arg);

    const T two_cos_thd3 = 2.0 * cos_thd3 * sign(rr);

    T eval2 = __ifelse(c2Negative, two_cos_thd3 / sqrtThreeOverA, 1.0);
    
    const Vector<T, M, 3> crow0({cxx - eval2, cxy, czx});
    const Vector<T, M, 3> crow1({cxy, cyy - eval2, cyz});
    const Vector<T, M, 3> crow2({czx, cyz, czz - eval2});

    //
    // do QR decomposition with column pivoting
    //
    const T k0 = crow0(0) * crow0(0) + cxy_cxy             + czx_czx;
    const T k1 = cxy_cxy             + crow1(1) * crow1(1) + cyz_cyz;
    const T k2 = czx_czx             + cyz_cyz             + crow2(2) * crow2(2);

    // returns zero or nan
    const bool k0gk1 = k1 <= k0;
    const bool k0gk2 = k2 <= k0;
    const bool k1gk2 = k2 <= k1;
    
    const bool k0_largest = k0gk1 & k0gk2;
    const bool k1_largest = k1gk2 & (!k0gk1);
    const bool k2_largest = !(k0_largest | k1_largest);
    const Vector<bool, 0, 3> k_largest({k0_largest, k1_largest, k2_largest});

    const T k_row1_0 = __ifelse(k0_largest, crow0(0), 0.0) +   
                       __ifelse(k1_largest, crow1(0), 0.0) + 
                       __ifelse(k2_largest, crow2(0), 0.0);
    const T k_row1_1 = __ifelse(k0_largest, crow0(1), 0.0) +
                       __ifelse(k1_largest, crow1(1), 0.0) +
                       __ifelse(k2_largest, crow2(1), 0.0);
    const T k_row1_2 = __ifelse(k0_largest, crow0(2), 0.0) +
                       __ifelse(k1_largest, crow1(2), 0.0) +
                       __ifelse(k2_largest, crow2(2), 0.0);
    const Vector<T, M, 3> k_row1({k_row1_0, k_row1_1, k_row1_2});
    const T row2_0 = __ifelse(k0_largest, crow1(0), crow0(0));
    const T row2_1 = __ifelse(k0_largest, crow1(1), crow0(1));
    const T row2_2 = __ifelse(k0_largest, crow1(2), crow0(2));
    Vector<T, M, 3> row2({row2_0, row2_1, row2_2});

    const T row3_0 = __ifelse(k2_largest, crow1(0), crow2(0));
    const T row3_1 = __ifelse(k2_largest, crow1(1), crow2(1));
    const T row3_2 = __ifelse(k2_largest, crow1(2), crow2(2));
    Vector<T, M, 3> row3({row3_0, row3_1, row3_2});

    const T ki_ki = 1.0 / ( 
        __ifelse(k0_largest, k0, 0.0) +
        __ifelse(k1_largest, k1, 0.0) +
        __ifelse(k2_largest, k2, 0.0)
    );
    
    const T ki_dpr1 = ki_ki * (
        k_row1(0) * row2(0) + k_row1(1) * row2(1) + k_row1(2) * row2(2)
    );
    const T ki_dpr2 = ki_ki * (
        k_row1(0) * row3(0) + k_row1(1) * row3(1) + k_row1(2) * row3(2)
    );

    row2 = row2 - ki_dpr1 * k_row1;
    row3 = row3 - ki_dpr2 * k_row1;

    const T a0 = row2(0) * row2(0) + row2(1) * row2(1) + row2(2) * row2(2);
    const T a1 = row3(0) * row3(0) + row3(1) * row3(1) + row3(2) * row3(2);

    const bool a0lea1 = a0 <= a1;

    const Vector<T, M, 3> a_row2 = __ifelse(a0lea1, row3, row2);
    const T ai_ai = 1.0 / __ifelse(a0lea1, a1, a0);

    Vector<T, M, 3> evec2({
        k_row1(1) * a_row2(2) - k_row1(2) * a_row2(1),
        k_row1(2) * a_row2(0) - k_row1(0) * a_row2(2),
        k_row1(0) * a_row2(1) - k_row1(1) * a_row2(0)
    });

    const T k_atr11 = cxx * k_row1(0) + cxy * k_row1(1) + czx * k_row1(2);
    const T k_atr21 = cxy * k_row1(0) + cyy * k_row1(1) + cyz * k_row1(2);
    const T k_atr31 = czx * k_row1(0) + cyz * k_row1(1) + czz * k_row1(2);

    const T a_atr12 = cxx * a_row2(0) + cxy * a_row2(1) + czx * a_row2(2);
    const T a_atr22 = cxy * a_row2(0) + cyy * a_row2(1) + cyz * a_row2(2);
    const T a_atr32 = czx * a_row2(0) + cyz * a_row2(1) + czz * a_row2(2);

    T rm2xx = (k_row1(0) * k_atr11 + k_row1(1) * k_atr21 + k_row1(2) * k_atr31) * ki_ki;
    const T k_a_rm2xy = (k_row1(0) * a_atr12 + k_row1(1) * a_atr22 + k_row1(2) * a_atr32);
    T rm2yy = (a_row2(0) * a_atr12 + a_row2(1) * a_atr22 + a_row2(2) * a_atr32) * ai_ai;
    const T rm2xy_rm2xy = k_a_rm2xy * k_a_rm2xy * ai_ai * ki_ki;

    //
    // Wilkinson shift
    //
    const T b = 0.5 * (rm2xx - rm2yy);

    const T sqrtTerm = sqrt(b * b + rm2xy_rm2xy) * sign(b);
    
    T eval0 = rm2yy + b - sqrtTerm;
    T eval1 = rm2xx + rm2yy - eval0;

    rm2xx -= eval0;
    rm2yy -= eval0;

    const T rm2xx2 = rm2xx * rm2xx;
    const T rm2yy2 = rm2yy * rm2yy;

    const T fac1 = static_cast<double>(__ifelse(rm2xx2 < rm2yy2, k_a_rm2xy * ai_ai, rm2xx));
    const T fac2 = static_cast<double>(__ifelse(rm2xx2 < rm2yy2, rm2yy, ki_ki * k_a_rm2xy));

    Vector<T, M, 3> evec0 = fac1 * a_row2 - fac2 * k_row1;

    const bool rm2xx2iszero = rm2xx2 == 0.0;
    const bool rm2xy_rm2xyiszero = rm2xy_rm2xy == 0.0;
    const bool both_zero = rm2xx2iszero && rm2xy_rm2xyiszero;

    // # check degeneracy
    
    evec0 = __ifelse(both_zero, a_row2, evec0);

    Vector<T, M, 3> evec1({
        evec2(1) * evec0(2) - evec2(2) * evec0(1),
        evec2(2) * evec0(0) - evec2(0) * evec0(2),
        evec2(0) * evec0(1) - evec2(1) * evec0(0)
    });

    eval0 = eval0 + c1;
    eval1 = eval1 + c1;
    eval2 = eval2 + c1;
    
    const T c2tol = (c1 * c1) * (-1.0e-30);

    const bool c2lsmall_neg = c2 < c2tol;
    
    eval0 = __ifelse(c2lsmall_neg, eval0, c1);
    eval1 = __ifelse(c2lsmall_neg, eval1, c1);
    eval2 = __ifelse(c2lsmall_neg, eval2, c1);

    evec0 = __ifelse(c2lsmall_neg, evec0, Vector<T, M, 3>({1.0, 0.0, 0.0}));
    evec1 = __ifelse(c2lsmall_neg, evec1, Vector<T, M, 3>({0.0, 1.0, 0.0}));
    evec2 = __ifelse(c2lsmall_neg, evec2, Vector<T, M, 3>({0.0, 0.0, 1.0}));
    
    
    Vector<T, M, 3> evals({eval0, eval1, eval2});
    Tensor2<T, M, 3> evecs;
    for (int i = 0; i < 3; ++i) {
        // evecs(0, i) = evec0(i);
        // evecs(1, i) = evec1(i);
        // evecs(2, i) = evec2(i);
        evecs(i, 0) = evec0(i);
        evecs(i, 1) = evec1(i);
        evecs(i, 2) = evec2(i);
    }

    if (evals(2) < evals(1)) {
        __swap(evals(1), evals(2));
        __swap(evecs, 1, 2);
    }
    if (evals(2) < evals(0)) {
        __swap(evals(0), evals(2));
        __swap(evecs, 0, 2);
    }
    if (evals(1) < evals(0)) {
        __swap(evals(0), evals(1));
        __swap(evecs, 0, 1);
    }

    return std::pair<Vector<T, M, 3>, Tensor2<T, M, 3>>(evals, evecs);
}

template<Scalar T, int M>
std::pair<Vector<T, M, 3>, Tensor2<T, M, 3>> eigen(const SymmetricTensor2<T, M, 3>& A) {
    T cmax = norm_inf(A);
    T cmaxInv = __ifelse(cmax > 0.0, 1.0 / cmax, 1.0);
    SymmetricTensor2<T, M, 3> scaledA = cmaxInv * A;
    auto [evals, evecs] = eigen_non_nomalized(scaledA);

    // TODO clean up below
    // we can remove some of the norm calls in the loop
    Vector<T, M, 3> norm_evec0({evecs(0, 0), evecs(1, 0), evecs(2, 0)});
    Vector<T, M, 3> norm_evec1({evecs(0, 1), evecs(1, 1), evecs(2, 1)});
    Vector<T, M, 3> norm_evec2({evecs(0, 2), evecs(1, 2), evecs(2, 2)});
    // Vector<T, M, 3> norm_evec0({evecs(0, 0), evecs(0, 1), evecs(0, 2)});
    // Vector<T, M, 3> norm_evec1({evecs(1, 0), evecs(1, 1), evecs(1, 2)});
    // Vector<T, M, 3> norm_evec2({evecs(2, 0), evecs(2, 1), evecs(2, 2)});

    for (int i = 0; i < 3; ++i) {
        evecs(i, 0) = evecs(i, 0) / ttk::norm_2(norm_evec0);
        evecs(i, 1) = evecs(i, 1) / ttk::norm_2(norm_evec1);
        evecs(i, 2) = evecs(i, 2) / ttk::norm_2(norm_evec2);
        // evecs(0, i) = evecs(0, i) / ttk::norm_2(norm_evec0);
        // evecs(1, i) = evecs(1, i) / ttk::norm_2(norm_evec1);
        // evecs(2, i) = evecs(2, i) / ttk::norm_2(norm_evec2);
    }
    
    evals = cmax * evals;

    return std::pair<Vector<T, M, 3>, Tensor2<T, M, 3>>(evals, evecs);
}

template<Scalar T, int M>
SymmetricTensor2<T, M, 3> from_eigen(const Vector<T, M, 3>& evals, const Tensor2<T, M, 3>& evecs) {
    SymmetricTensor2<T, M, 3> A;
    A.fill(0.0);
    for (int i = 0; i < 3; ++i) {
        // form N \otimes N where N is the column vector of evecs(:, i)
        const Vector<T, M, 3> N({evecs(0, i), evecs(1, i), evecs(2, i)});
        const SymmetricTensor2<T, M, 3> NoN = otimes(N);
        A = A + evals(i) * NoN;
    }
    return A;
}

} // end namespace ttk
