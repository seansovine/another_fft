// top-level library module

pub mod basic_ops;
pub mod fft;

use another_fft::fft;
use image::{ImageBuffer, Rgba};

// simple test

pub fn fft_test() {
  let mut test_array: [f64; 16] = [
    1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64, 1f64, 0f64,
  ];

  fft(&mut test_array, false);
  println!("FFT of array: {:?}", test_array);
}

// borrowed from wgpu_grapher

pub struct Image {
  pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
  pub dimensions: (u32, u32),
}

impl Image {
  pub fn from_file(filepath: &str) -> Self {
    // TODO: handle error vs panicking
    let image_bytes = std::fs::read(filepath)
      .unwrap_or_else(|_| panic!("Unable to read image at path: {}", filepath));

    let image = image::load_from_memory(&image_bytes).unwrap().to_rgba8();
    let dimensions = image.dimensions();

    Self { image, dimensions }
  }
}
