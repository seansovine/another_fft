// Home of functions to compute the FFT.

use std::f64::consts::PI;

/// This is the implementation from
///
/// ```text
/// Press, et al., _Numerical Reciples_, Third Edition.
/// Cambridge University Press, 2007.
/// ```
///
/// which the authors cite as originating in the work of
/// N.M. Brenner. (See Rader and Brenner, 1976.)
///
/// __Arguments:__
///
/// + `data` - the sequence of N complex values with real parts in
///   even-numbered entries and imaginary parts in odd entries.
///
/// + `inverse` - if true will compute the inverse transform;
///   the convention here is to put the 1/N normalizing factor
///   on the inverse transform.
///
pub fn fft(data: &mut [f64], inverse: bool) {
  let data_len = data.len() / 2;
  if (data_len as f64).log2().fract() != 0.0 {
    panic!("FFT routine is only implemented for arrays with length a power of 2.")
  }

  let n = data.len() / 2;
  let nn = n << 1;

  // phase sign and normalizing factor for inverse transform
  let inv_sign: f64 = if inverse { 1f64 } else { -1f64 };
  let inv_mult = if inverse { 1f64 / (n as f64) } else { 1f64 };

  // Bit reversal portion:
  //
  //  This algorithm swaps each complex data entry (pair of slice entries)
  //  with the entry whose index has the binary representation that is the
  //  reverse of its binary representation.

  let mut j: usize = 1;
  let mut m: usize;

  for i in (1..nn).step_by(2) {
    if j > i {
      data.swap(j - 1, i - 1);
      data.swap(j, i);
    }

    // compute next potential swap row
    m = n;
    while m >= 2 && j > m {
      j -= m;
      m >>= 1;
    }
    j += m;

    // divide by complex sequence length if inverse
    data[i - 1] *= inv_mult;
    data[i] *= inv_mult;
  }

  // Danielson-Lanczos iteration:
  //
  //  The next part performs the "butterfly" operations on the vector
  //  that was resorted in the previous step. A recurrence is used to
  //  efficiently compute the trigonmetric functions involved.

  let mut mmax: usize = 2;
  let mut istep: usize;

  let mut theta: f64;
  let mut wtemp: f64;
  let mut wpr: f64;
  let mut wpi: f64;

  while nn > mmax {
    istep = mmax << 1;

    theta = inv_sign * 2f64 * PI / (mmax as f64);
    wtemp = (theta / 2f64).sin();

    wpr = -2.0f64 * wtemp * wtemp;
    wpi = theta.sin();

    // start recurrence for trigonometric functions of whole multiples of a given angle
    let mut wr: f64 = 1.0;
    let mut wi: f64 = 0.0;

    let mut tempr: f64;
    let mut tempi: f64;

    for m in (1..mmax).step_by(2) {
      for i in (m..=nn).step_by(istep) {
        j = i + mmax;
        tempr = wr * data[j - 1] - wi * data[j];
        tempi = wr * data[j] + wi * data[j - 1];

        data[j - 1] = data[i - 1] - tempr;
        data[j] = data[i] - tempi;
        data[i - 1] += tempr;
        data[i] += tempi;
      }

      wtemp = wr;
      wr = wr * wpr - wi * wpi + wr;
      wi = wi * wpr + wtemp * wpi + wi;
    }

    mmax = istep;
  }
}

// 2D FFT implementation

/// Basic implementation that works by applying the row-wise fft, transposing
/// and applying the row-wise fft again, and then transposing again.
///
/// This version allocates a temporary array as part of the transposition, but
/// a clever in-place algorithm for the transposition of a flattened array may
/// exist.
///
/// __Arguments:__
///
/// + `data` - flattened 2D array of (real, complex) pairs
///
/// + `dimensions` - (width, height) of array of _complex_ values
///
pub fn fft_2d(data: &mut [f64], dimensions: (usize, usize), inverse: bool) {
  // width in terms of conceptual complex data
  // entries; data width is twice this number
  let width = dimensions.0;
  let height = dimensions.1;

  // row-wise fft
  for i in 0..height {
    let offset = i * width * 2;
    fft(&mut data[offset..offset + width * 2], inverse);
  }

  // we just allocate an intermediate array; a clever in-place algorithm for
  // transposing a matrix in flattened form may exist, but this works for now
  let mut transposed: Vec<f64> = vec![0.0_f64; 2 * width * height];

  complex_transpose(data, dimensions, &mut transposed);
  // dimensions are now reversed

  // column-wise fft (row-wised on transposed matrix)
  for i in 0..width {
    let offset = i * height * 2;
    fft(&mut transposed[offset..offset + height * 2], inverse);
  }

  // Transpose back now.
  complex_transpose(&transposed, (dimensions.1, dimensions.0), data);
}

/// Transpose a matrix of complex entries represented as double-width
/// matrix of real entries.
///
/// __Note:__
///
/// There is possibly a clever way to do the transposed on the flattened
/// array in-place, but here we just allocate a new array for simplicity.
///
/// For our use cases there should be plenty of memory to spare, and we don't
/// need that much efficiency that allocation will be too slow.
pub fn complex_transpose(data_in: &[f64], in_dimensions: (usize, usize), data_out: &mut [f64]) {
  let width = in_dimensions.0;
  let height = in_dimensions.1;

  for i in 0..height {
    for j in 0..width {
      // original and new real-part indices
      let n_r = 2 * (i * width + j);
      let n_r_new = (j * height + i) * 2;

      // swap real parts
      data_out[n_r_new] = data_in[n_r];
      // swap complex parts -
      // source and destination adjacent to real parts
      data_out[n_r_new + 1] = data_in[n_r + 1];
    }
  }
}

// Unit tests.

#[cfg(test)]
mod tests {
  use super::*;

  const TEST_ARRAY: [f64; 12] = [
    1.0, 6.0, 2.0, 5.0, 3.0, 4.0, //
    4.0, 3.0, 5.0, 2.0, 6.0, 1.0, //
  ];

  const TEST_ARRAY_WIDTH: usize = 3;
  const TEST_ARRAY_HEIGHT: usize = 2;

  #[test]
  fn complex_transpose_test() {
    println!("Original matrix:");
    print_matrix(&TEST_ARRAY, TEST_ARRAY_WIDTH, TEST_ARRAY_HEIGHT);

    let mut transposed: Vec<f64> = vec![0.0_f64; 2 * TEST_ARRAY_WIDTH * TEST_ARRAY_HEIGHT];
    complex_transpose(&TEST_ARRAY, (3, 2), &mut transposed);

    println!("\nTransposed matrix:");
    print_matrix(&transposed, TEST_ARRAY_HEIGHT, TEST_ARRAY_WIDTH);
  }

  const FFT_TEST_ARRAY: [f64; 16] = [
    1.0, 6.0, 2.0, 5.0, 3.0, 4.0, 0.0, 0.0, //
    4.0, 3.0, 5.0, 2.0, 6.0, 1.0, 0.0, 0.0, //
  ];

  const FFT_TEST_ARRAY_WIDTH: usize = 4;
  const FFT_TEST_ARRAY_HEIGHT: usize = 2;

  #[test]
  fn fft_2d_test() {
    let mut test_vec = Vec::from(FFT_TEST_ARRAY);
    println!("Test matrix:");
    print_matrix(&test_vec, FFT_TEST_ARRAY_WIDTH, FFT_TEST_ARRAY_HEIGHT);

    let mut inverse = false;
    fft_2d(
      &mut test_vec,
      (FFT_TEST_ARRAY_WIDTH, FFT_TEST_ARRAY_HEIGHT),
      inverse,
    );

    println!("\nFFT of test matrix:");
    print_matrix(&test_vec, FFT_TEST_ARRAY_WIDTH, FFT_TEST_ARRAY_HEIGHT);

    inverse = true;
    fft_2d(
      &mut test_vec,
      (FFT_TEST_ARRAY_WIDTH, FFT_TEST_ARRAY_HEIGHT),
      inverse,
    );

    println!("\nIFFT of FFT of test matrix:");
    print_matrix(&test_vec, FFT_TEST_ARRAY_WIDTH, FFT_TEST_ARRAY_HEIGHT);
  }

  // test helper functions

  fn print_matrix(data: &[f64], width: usize, height: usize) {
    for i in 0..height {
      for j in 0..width {
        let offset = (i * width + j) * 2;
        let r = data[offset];
        let c = data[offset + 1];
        print!("({:>4.1}, {:>4.1}) ", r, c);
      }
      println!();
    }
  }
}
