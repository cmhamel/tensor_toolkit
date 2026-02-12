#pragma once

namespace ttk {

// index metrics
// NOTE this one is probably likely unnecesssary
// since memory is contiguous/linear here
// but added here for completeness so
// its absence does not confuse people
enum VecIndices {
    _V_0 = 0,
    _V_1 = 1,
    _V_2 = 2
};

enum Tensor2Indices {
    _T2_00 = 0, _T2_01 = 3, _T2_02 = 4,
    _T2_10 = 6, _T2_11 = 1, _T2_12 = 5,
    _T2_20 = 7, _T2_21 = 8, _T2_22 = 2
};

enum SymmetricTensor2Indices {
    _ST2_00 = 0, _ST2_01 = 3, _ST2_02 = 4,
    _ST2_10 = 3, _ST2_11 = 1, _ST2_12 = 5,
    _ST2_20 = 4, _ST2_21 = 5, _ST2_22 = 2
};

enum Metrics {
    CARTESIAN = 0
    // TODO add more
};

enum Symmetry {
    FULL = 0,
    SYMM = 1,
    SKEW = 2
};

} // end namespace ttk
