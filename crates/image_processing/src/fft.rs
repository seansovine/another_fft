// Initial code for visualizing the FFT of a grayscale image.
//
// We won't take pains to make anything in the initial version of this
// particularly efficient; we're going to start by just getting it to work,
// then we'll optimize later.

use another_fft::fft_2d;

use image::{ImageBuffer, Rgba};

use crate::Image;

pub fn fft_image(in_path: &str) {
  let image = Image::from_file(in_path);
  let dims = image.dimensions;

  println!("Image width and height: ({}, {})", dims.0, dims.1);
  println!("Converting image to FFT format.");

  let mut grayscale_data =
    GrayscaleImageData::from_rgba_bytes(&image.image, (dims.0 as usize, dims.1 as usize));

  println!("Performing FFT.");

  fft_2d(
    &mut grayscale_data.complex_data,
    grayscale_data.dimensions,
    false,
  );

  println!("Converting FFT to image format.");

  let fft_image = grayscale_data.to_fft_image_buffer(true);

  println!("Writing image to file.");

  const FFT_OUTPATH: &str = "test_data/fft.jpg";
  fft_image
    .save_with_format(FFT_OUTPATH, image::ImageFormat::Jpeg)
    .unwrap();

  println!("Performing inverse FFT.");

  fft_2d(
    &mut grayscale_data.complex_data,
    grayscale_data.dimensions,
    true,
  );

  println!("Writing reconstructed image to file.");

  let ifft_image = grayscale_data.to_fft_image_buffer(false);

  const RECONSTRUCTED_OUTPATH: &str = "test_data/reconstructed.jpg";
  ifft_image
    .save_with_format(RECONSTRUCTED_OUTPATH, image::ImageFormat::Jpeg)
    .unwrap();
}

/// Represents the raw data of a grayscale image as an array
/// of complex floats, for use as input to the FFT function.
#[derive(Debug)]
struct GrayscaleImageData {
  complex_data: Vec<f64>,
  dimensions: (usize, usize),
}

impl GrayscaleImageData {
  fn from_rgba_bytes(
    img_data: &ImageBuffer<Rgba<u8>, Vec<u8>>,
    dimensions: (usize, usize),
  ) -> Self {
    let width = dimensions.0;
    let height = dimensions.1;

    // fft expects flat array of interleaved real and complex parts
    let mut complex_data: Vec<f64> = vec![0.0; 2 * width * height];

    // we assume the pixels are grayscale, so we only read from the first channel
    for i in 0..height {
      for j in 0..width {
        let offset = 2 * (width * i + j);
        let pixel = img_data[(j as u32, i as u32)].0;
        complex_data[offset] = pixel[0] as f64;
      }
    }

    Self {
      complex_data,
      dimensions,
    }
  }

  fn to_fft_image_buffer(&self, shift_and_scale: bool) -> ImageBuffer<Rgba<u8>, Vec<u8>> {
    let width = self.dimensions.0;
    let height = self.dimensions.1;

    let mut image = image::RgbaImage::new(width as u32, height as u32);

    let (scale_factor, j_shift, i_shift) = if shift_and_scale {
      (1000.0_f64, width / 2, height / 2)
    } else {
      (1.0_f64, 0, 0)
    };

    for i in 0..height {
      for j in 0..width {
        let data_offset = 2 * (i * width + j);
        let fft_r = self.complex_data[data_offset] / scale_factor;
        let fft_c = self.complex_data[data_offset + 1] / scale_factor;

        // magnitude of complex entry clamped to [0, 255]
        let pixel_val = (fft_r.powi(2) + fft_c.powi(2)).sqrt().clamp(0.0, 255.0) as u8;

        // println!("FFT val: {:.3}, {:.3}", fft_r, fft_c);
        // println!("Pixel val: {}", pixel_val);

        // maybe move origin to image center
        let j_shifted = (j + j_shift) % width;
        let i_shifted = (i + i_shift) % height;

        image.put_pixel(
          j_shifted as u32,
          i_shifted as u32,
          image::Rgba([pixel_val, pixel_val, pixel_val, 255]),
        );
      }
    }

    image
  }
}
