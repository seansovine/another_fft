// This file is a place to test the FFT implementation(s).

use another_fft::fft;
use ndarray::Array1;
use std::f64::consts::PI;

/// Which test sequence to use.
const TEST: usize = 3;

use clap::Parser;

#[derive(Parser)]
#[command(about, long_about = None)]
struct Args {
  #[arg(long)]
  test_number: Option<usize>,
}

/// Takes the FFT of a test sequence, followed by the inverse FFT.
/// Outputs the original sequence, the FFT, and the IFFT of the FFT.
fn main() {
  let args = Args::parse();
  let test = match args.test_number {
    Some(n) => n,
    None => TEST,
  };

  let mut test_data = match test {
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

fn get_test_array_3() -> Array1<f64> {
  Array1::<f64>::from_iter([
    1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64,
  ])
}

fn get_test_array_2() -> Array1<f64> {
  Array1::<f64>::from_iter([1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64])
}

fn get_test_array_1() -> Array1<f64> {
  // represents complex number as sequence (real, complex, ...)
  let mut data = Array1::<f64>::zeros([(2_usize).pow(2 + 1)]);
  let mut_slice = data.as_slice_mut().unwrap();

  #[allow(non_snake_case)]
  let N = (mut_slice.len() / 2) as f64;
  let k = 1f64;

  // initialize array as kth harmonic for testing
  for n in (0..mut_slice.len()).step_by(2) {
    let phase = 2f64 * PI * (k / N) * ((n / 2) as f64);

    mut_slice[n] = phase.cos();
    mut_slice[n + 1] = phase.sin();
  }

  data
}
