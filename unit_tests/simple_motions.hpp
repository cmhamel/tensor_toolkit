// #pragma once
// #include <tensor_toolkit.hpp>

// template<typename T, int M, int D>
// class UniaxialStrain {
// public:
//     TTK_INLINE
//     constexpr UniaxialStrain() = default;
//     ~UniaxialStrain() = default;

//     TTK_INLINE
//     ttk::Tensor2<T, M, D>  deformationGradient(double stretch) const {
//         ttk::Tensor2<T, M, D> F = ttk::identity<ttk::Tensor2<T, M, D>>();
//         F(0, 0) = stretch;
//         return F;
//     }

//     TTK_INLINE
//     ttk::Tensor2<T, M, D> displacementGradient(double stretch) const {
//         return deformationGradient(stretch) - ttk::identity<ttk::Tensor2<T, M, D>>();
//     }
// };
