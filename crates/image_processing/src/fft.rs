// Code for computing the FFT of an image and possibly applying a Fourier-side filter.
//
// The computation is parallelized with Rayon, but there is more
// work that could be done to optimize and improve efficiency.

use another_fft::fft_2d_para;

use image::{ImageBuffer, Rgba};
use std::time;

use crate::ImageProcessor;

/// Compute image FFT, possibly apply Fourier-side filter, and then recover
/// the image by applying IFFT.
///
/// Note: Assumes image dimensions are powers of 2.
pub fn fft_image(image: &ImageProcessor, filter: bool) -> Result<(), String> {
    let dims = image.dimensions;
    let pool = &image.thread_pool;

    println!("> Image width and height: ({}, {})", dims.0, dims.1);

    let report_elapsed = |time: time::Instant| {
        let elapsed = time.elapsed().as_secs_f32();
        println!("... {:>2.3}s", elapsed);
    };

    //// setup for parallel FFT

    println!("Doing initial setup for parallel FFT.");

    let time = time::Instant::now();
    // pre-allocate working space for the parallel FFT code
    let mut working_buffer: Vec<f64> = vec![0.0_f64; 2 * dims.0 as usize * dims.1 as usize];

    report_elapsed(time);

    //// forward fft

    println!("Converting image to FFT format.");

    let time = time::Instant::now();
    let mut grayscale_data =
        GrayscaleImageData::from_rgba_bytes(&image.image, (dims.0 as usize, dims.1 as usize));

    report_elapsed(time);
    println!("Performing FFT.");

    let time = time::Instant::now();
    fft_2d_para(
        &mut grayscale_data.complex_data,
        &mut working_buffer,
        grayscale_data.dimensions,
        false,
        pool,
    );

    report_elapsed(time);
    println!("Converting FFT to image format.");

    let time = time::Instant::now();
    let fft_image = grayscale_data.to_fft_image_buffer(true);

    report_elapsed(time);
    println!("Writing image to file.");

    const FFT_OUTPATH: &str = "test_data/fft.jpg";

    let time = time::Instant::now();
    fft_image
        .save_with_format(FFT_OUTPATH, image::ImageFormat::Jpeg)
        .unwrap();

    report_elapsed(time);

    //// maybe apply filter on Fourier side

    if filter {
        println!("Applying filter to Fourier coefficients.");

        let time = time::Instant::now();

        // annular frequency domain cutoff
        #[allow(unused)]
        let annular_cutoff = |i: usize, j: usize| -> (f64, f64) {
            let mag_sq = i.pow(2) + j.pow(2);

            if 50_usize.pow(2) <= mag_sq && mag_sq <= 500_usize.pow(2) {
                return (1.0, 0.0);
            }

            (0.0, 0.0)
        };

        // linearly boost higher frequencies for sharpening effect
        let linear_sharpen = |i: usize, j: usize| -> (f64, f64) {
            let mag_sq = i.pow(2) + j.pow(2);

            if 500_usize.pow(2) <= mag_sq {
                return (((mag_sq as f64).sqrt() / 1000.0 + 1.0), 0.0);
            }

            (1.0, 0.0)
        };

        grayscale_data.apply_filter_generic(linear_sharpen);

        report_elapsed(time);
        println!("Converting filtered FFT to image format.");

        let time = time::Instant::now();
        let fft_image = grayscale_data.to_fft_image_buffer(true);

        report_elapsed(time);
        println!("Writing image to file.");

        const FILTERED_FFT_OUTPATH: &str = "test_data/fft_filtered.jpg";

        let time = time::Instant::now();
        fft_image
            .save_with_format(FILTERED_FFT_OUTPATH, image::ImageFormat::Jpeg)
            .unwrap();

        report_elapsed(time);
    }

    //// inverse fft of fft

    println!("Performing inverse FFT.");

    let time = time::Instant::now();
    fft_2d_para(
        &mut grayscale_data.complex_data,
        &mut working_buffer,
        grayscale_data.dimensions,
        true,
        pool,
    );

    report_elapsed(time);
    println!("Converting data back to grayscale image format.");

    let time = time::Instant::now();
    let ifft_image = grayscale_data.to_fft_image_buffer(false);

    report_elapsed(time);
    println!("Writing reconstructed image to file.");

    const RECONSTRUCTED_OUTPATH: &str = "test_data/reconstructed.jpg";

    let time = time::Instant::now();
    ifft_image
        .save_with_format(RECONSTRUCTED_OUTPATH, image::ImageFormat::Jpeg)
        .unwrap();

    report_elapsed(time);

    Ok(())
}

// Image Fourier data structure.

/// Represents the raw data of a grayscale image as an array
/// of complex floats, for use as input to the FFT function.
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

        // our 2d fft expects a flat array of interleaved real and complex parts
        let mut complex_data: Vec<f64> = vec![0.0; 2 * width * height];

        for i in 0..height {
            for j in 0..width {
                let offset = 2 * (width * i + j);
                let pixel = img_data[(j as u32, i as u32)].0;

                // we assume the pixels are grayscale,
                // so we only read from the first channel
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

        // set non-trivial scale factor and origin
        // shift if shift_and_scale is true
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

    /// For now we implement a high-pass filter; we can implement other
    /// filter types later and then make this function more generic.
    #[allow(unused)]
    fn apply_filter(&mut self, cutoff_radius: f32) {
        let width = self.dimensions.0;
        let height = self.dimensions.1;

        let cutoff_radius_sqr: f32 = cutoff_radius.powi(2);

        // apply circular cutoff around Fourier origin
        for i in 0..height {
            for j in 0..width {
                // we want distance from 0 including wraparound
                let i_signed = ((i + height / 2) % height) as f32 - height as f32 / 2.0;
                let j_signed = ((j + width / 2) % width) as f32 - width as f32 / 2.0;

                let mag_sqr = (i_signed).powi(2) + (j_signed).powi(2);

                if mag_sqr <= cutoff_radius_sqr {
                    let offset = 2 * (i * width + j);
                    self.complex_data[offset] = 0.0;
                    self.complex_data[offset + 1] = 0.0;
                }
            }
        }
    }

    /// Apply filter_fn to the fourier coefficients held by this struct.
    fn apply_filter_generic<F>(&mut self, filter_fn: F)
    where
        F: Fn(usize, usize) -> (f64, f64),
    {
        let width = self.dimensions.0;
        let height = self.dimensions.1;

        // apply circular cutoff around Fourier origin
        for i in 0..height {
            for j in 0..width {
                // we want distance from 0 including wraparound
                let i_signed = ((i + height / 2) % height) - height / 2;
                let j_signed = ((j + width / 2) % width) - width / 2;

                let offset = 2 * (i * width + j);

                let re = self.complex_data[offset];
                let im = self.complex_data[offset + 1];

                let (re_f, im_f) = filter_fn(i_signed, j_signed);

                self.complex_data[offset] = re * re_f - im * im_f;
                self.complex_data[offset + 1] = re * im_f + im * re_f;
            }
        }
    }
}
