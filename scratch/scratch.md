/// Transpose a matrix of complex entries represented as double-width matrix of real entries.
/// There is possibly a clever way to do the transposed on the flattened array in-place,
/// but here we just allocate a new array for simplicity. For our use cases there should
/// be plenty of memory to spare, and we don't need that much efficiency that allocation will
/// be too slow. for our use case.
