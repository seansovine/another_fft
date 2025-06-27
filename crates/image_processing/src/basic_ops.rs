// Basic image operations like resize, grayscale.

use crate::ImageProcessor;

/// Use the image crate to resize image.
pub fn resize(image: &ImageProcessor, new_width: usize, new_height: usize) -> Result<(), String> {
    println!("Image loaded successfully!");
    println!("Dimensions: {:?}", image.dimensions);

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

    Ok(())
}

/// Convert image pixels to grayscale but still in RGBA format.
pub fn to_grayscale(mut image: ImageProcessor) -> Result<(), String> {
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

    Ok(())
}
