#pragma once

#ifdef TENSOR_TOOLKIT_ENABLE_ENZYME

#include <enzyme/enzyme>

namespace ttk {

// TODO lets make some nice wrappers...

// enzyme tags, from enzyme docs
// int enzyme_const;
// int enzyme_dup;
// int enzyme_out;

// some helpers
template<typename... T>
void __enzyme_autodiff(void*, T...);
template<typename... T>
void __enzyme_fwddiff(void*, T...);

// // helper methods
// template<typename Obj, typename... Args>
// void gradient(void* f, Obj* obj, Args... args) {
//     static_assert(
//         sizeof...(Args) % 2 == 0,
//         "Arguments must come in (primal, tangent) pairs"
//     );

//     for (std::size_t i = 0; i < args.size(); i += 2) {
//         enzyme::Duplicated<T> dup{*args[i], *args[i + 1]};
//         // use(dup);
//     }
// }

} // end namespace ttk

#endif
