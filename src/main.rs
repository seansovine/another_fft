/// Implementing the most basic FFT on nice Rust data structures.
///
///
use another_fft::fft;

use ndarray::Array1;

use std::f64::consts::PI;

const TEST: usize = 1;

fn main() {
  let mut test_data = match TEST {
    1 => get_test_array_1(),
    2 => get_test_array_2(),
    3 => get_test_array_3(),
    _ => unimplemented!(),
  };

  println!("Original:  {:#?}", test_data);

  let mut inverse = false;
  fft(test_data.as_slice_mut().unwrap(), inverse);
  println!("FT:        {:#?}", test_data);

  inverse = true;
  fft(test_data.as_slice_mut().unwrap(), inverse);
  println!("IFT of FT: {:#?}", test_data);
}

#[allow(unused)]
fn get_test_array_3() -> Array1<f64> {
  Array1::<f64>::from_iter([
    1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64,
  ])
}

#[allow(unused)]
fn get_test_array_2() -> Array1<f64> {
  Array1::<f64>::from_iter([1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64])
}

#[allow(unused)]
fn get_test_array_1() -> Array1<f64> {
  // Represents complex number as sequence (real, complex, ...).
  let mut data = Array1::<f64>::zeros([(2 as usize).pow(2 + 1)]);
  let mut_slice = data.as_slice_mut().unwrap();

  #[allow(non_snake_case)]
  let N = (mut_slice.len() / 2) as f64;
  let k = 1f64;

  // Initialize array as kth harmonic for testing.
  for n in (0..mut_slice.len()).step_by(2) {
    let phase = 2f64 * PI * (k / N) * ((n / 2) as f64);

    mut_slice[n] = phase.cos();
    mut_slice[n + 1] = phase.sin();
  }

  data
}
