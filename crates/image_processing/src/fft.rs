// Initial code for visualizing the FFT of a grayscale image.
//
// Nothing in the initial version of this will be particularly efficient;
// we're going to start by just getting it to work, then we'll optimize later.

/// Represents the raw data of a grayscale image as an array
/// of complex bytes, for use as input to the FFT function.
#[derive(Debug)]
struct _GrayscaleImageData {
  complex_bytes: Vec<u8>,
  dimensions: (usize, usize),
}

impl _GrayscaleImageData {
  // TODO: construct from image buffer bytes
}
