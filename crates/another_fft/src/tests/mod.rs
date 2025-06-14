// unit tests

use super::*;

const TEST_ARRAY: [f64; 12] = [
  1.0, 6.0, 2.0, 5.0, 3.0, 4.0, //
  4.0, 3.0, 5.0, 2.0, 6.0, 1.0, //
];

const TEST_ARRAY_WIDTH: usize = 3;
const TEST_ARRAY_HEIGHT: usize = 2;

// Note: These tests just write out results to the console for
// visual inspection for now. In particular, you'll want to run
// cargo test with the --show-output flag.

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
