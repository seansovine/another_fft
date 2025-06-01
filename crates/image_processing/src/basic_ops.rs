// Basic image operations like resize, grayscale.

use crate::Image;

// Use the image crate to resize an image.
//
// Initially we'll use this to resize an image to power of two dimensions
// so we can apply the 2-dimensional FFT to it.

pub fn resize(in_path: &str, new_width: usize, new_height: usize) {
  let image = Image::from_file(in_path);
  println!("Image loaded successfully!");
  println!("Dimensions: {:?}", image.dimensions);

  // let new_dimensions: (u32, u32) = (8192, 4096);

  println!("Resizing image...");

  let resize_filter = image::imageops::FilterType::CatmullRom;
  let buffer = image::imageops::resize(
    &image.image,
    new_width as u32,
    new_height as u32,
    resize_filter,
  );

  println!("Resize success!");

  // save resized image, to hard-coded file for now

  const OUTPATH: &str = "test_data/resized.jpg";
  buffer
    .save_with_format(OUTPATH, image::ImageFormat::Jpeg)
    .unwrap();

  // view result: wgpu_grapher image --path test/resized.jpg
}

// Convert image pixels to grayscale but still RGBA representation.

pub fn to_grayscale(in_path: &str) {
  let mut image = Image::from_file(in_path);
  let dims = image.dimensions;

  println!("Image loaded successfully!");
  println!("Dimensions: {:?}", image.dimensions);

  let to_grayscale = |p: &[u8]| -> u8 {
    ((p[0] as f64 * 0.299) + (p[1] as f64 * 0.587) + (p[2] as f64 * 0.114)) as u8
  };

  println!("Converting image to grayscale (RGBA)...");

  for i in 0..dims.0 {
    for j in 0..dims.1 {
      let pixel = &mut image.image[(i, j)].0;
      let grayscale_val = to_grayscale(pixel);

      pixel[0] = grayscale_val;
      pixel[1] = grayscale_val;
      pixel[2] = grayscale_val;
    }
  }

  println!("Conversion success!");

  // save grayscale image, to hard-coded file for now

  const OUTPATH: &str = "test_data/grayscale.jpg";
  image
    .image
    .save_with_format(OUTPATH, image::ImageFormat::Jpeg)
    .unwrap();

  // view result: wgpu_grapher image --path test/grayscale.jpg
}
