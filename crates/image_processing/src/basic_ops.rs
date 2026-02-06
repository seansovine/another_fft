// Basic image operations like resize, grayscale.

use image::{ImageBuffer, Rgba};

use crate::ImageProcessor;

/// Use the image crate to resize image.
pub fn save_resize(
    image_proc: &ImageProcessor,
    new_width: usize,
    new_height: usize,
) -> Result<(), String> {
    println!("Image loaded successfully!");
    println!("Dimensions: {:?}", image_proc.dimensions);

    println!("Resizing image...");

    let resize_filter = image::imageops::FilterType::CatmullRom;
    let buffer = image::imageops::resize(
        &image_proc.image,
        new_width as u32,
        new_height as u32,
        resize_filter,
    );

    println!("Resize success!");

    // save resized image, to hard-coded file for now

    buffer
        .save_with_format(&image_proc.output_path, image::ImageFormat::Jpeg)
        .unwrap();

    Ok(())
}

fn pixel_to_grayscale(p: &[u8]) -> u8 {
    ((p[0] as f64 * 0.299) + (p[1] as f64 * 0.587) + (p[2] as f64 * 0.114)) as u8
}

/// Convert image pixels to grayscale but still in RGBA format.
pub fn save_grayscale(mut image_proc: ImageProcessor) -> Result<(), String> {
    let dims = image_proc.dimensions;

    println!("Image loaded successfully!");
    println!("Dimensions: {:?}", image_proc.dimensions);

    println!("Converting image to grayscale (RGBA)...");

    for i in 0..dims.0 {
        for j in 0..dims.1 {
            let pixel = &mut image_proc.image[(i, j)].0;
            let grayscale_val = pixel_to_grayscale(pixel);

            pixel[0] = grayscale_val;
            pixel[1] = grayscale_val;
            pixel[2] = grayscale_val;
        }
    }

    println!("Conversion success!");

    // save grayscale image, to hard-coded file for now

    image_proc
        .image
        .save_with_format(&image_proc.output_path, image::ImageFormat::Jpeg)
        .unwrap();

    Ok(())
}

pub fn grayscale_bytes(image: &ImageBuffer<Rgba<u8>, Vec<u8>>) -> Vec<u8> {
    let mut out_buffer = vec![0; image.height() as usize * image.width() as usize];
    let image_width = image.width();
    for i in 0..image.height() {
        for j in 0..image_width {
            let pixel = &image[(j, i)].0;
            out_buffer[(i * image_width + j) as usize] = pixel_to_grayscale(pixel);
        }
    }
    out_buffer
}
