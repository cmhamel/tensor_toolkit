#pragma once

namespace ttk {

// helper to make things build correctly
// with certain compiler flags
template<typename>
inline constexpr bool __always_false = false;

} // end namespace ttk