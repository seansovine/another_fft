// Use the image crate to resize an image.
//
// Initially we'll use this to resize an image to power of two dimensions
// so we can apply the 2-dimensional FFT to it.

use crate::Image;

// we'll start with a hard-coded test, and then turn it into an API
pub fn resize(path: &str) {
  let image = Image::from_file(path);
  println!("Image loaded successfully!");
  println!("Dimensions: {:?}", image.dimensions);

  let new_dimensions: (u32, u32) = (8192, 4096);

  println!("Resizing image...");

  let resize_filter = image::imageops::FilterType::CatmullRom;
  let buffer = image::imageops::resize(
    &image.image,
    new_dimensions.0,
    new_dimensions.1,
    resize_filter,
  );

  println!("Resize success!");

  // save resized image, to hard-coded file for now

  const OUTPATH: &str = "test/resized.jpg";
  buffer
    .save_with_format(OUTPATH, image::ImageFormat::Jpeg)
    .unwrap();
}

// view result: wgpu_grapher image --path test/resized.jpg
