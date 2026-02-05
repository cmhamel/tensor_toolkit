#pragma once

namespace ttk {

// helper to make things build correctly
// with certain compiler flags
template<typename...>
inline constexpr bool __always_false = false;

// concept to help with checking
// if things are in the same frame
template<typename A, typename B>
concept SameFrame = std::is_same_v<A, B>;

} // end namespace ttk