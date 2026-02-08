// Code for implementing convolution filters on images.
//
// This version is not optimized.

use crate::{ImageProcessor, basic_ops::grayscale_bytes};

use clap::Subcommand;
use image::{ImageBuffer, Luma, Rgba};
use num_format::{Locale, ToFormattedString};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

use std::time;
use std::{
    ops::{Index, IndexMut},
    time::Instant,
};

// ---------------------------------------------
// Simple structure for 3x3 convolution kernels.

pub struct Kernel3X3 {
    m: [[f64; 3]; 3],
}

impl Kernel3X3 {
    pub fn sobel_x() -> Self {
        Self {
            m: [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        }
    }

    pub fn sobel_y() -> Self {
        Self {
            m: [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        }
    }

    pub fn avg() -> Self {
        Self {
            m: [[1.0 / 9.0; 3]; 3],
        }
    }
}

// -------------------------------------------------------------
// Things being tried out towards efficient convolution / Sobel.

/// An approach being tried out towards cache-efficient sobel implementation.
///
/// Assumptions: That image and output are stored row-major, with rows of `width` entries,
/// and that image contains two more rows than output.
///
/// Results: It isn't significantly faster than our main implementation, and it does less.
#[allow(dead_code)]
fn sobel_op_test_1(width: usize, image: &[u8], output: &mut [f64]) {
    assert!(width.is_multiple_of(CACHE_LINE_SIZE));
    assert!(image.len() / width == output.len() / width + 2);

    let m_x = Kernel3X3::sobel_x().m;
    let m_y = Kernel3X3::sobel_y().m;

    let mut buffer = vec![0f64; output.len()];

    // Top rows of input only contributes to two output rows.
    for j in 0..width {
        output[j + 1] += m_x[0][0] * image[j] as f64
            + m_x[0][1] * image[j + 1] as f64
            + m_x[0][2] * image[j + 2] as f64;
        buffer[j + 1] += m_y[0][0] * image[j] as f64
            + m_y[0][1] * image[j + 1] as f64
            + m_y[0][2] * image[j + 2] as f64;
    }
    for j in width..2 * width {
        output[j + 1] += m_x[0][0] * image[j] as f64
            + m_x[0][1] * image[j + 1] as f64
            + m_x[0][2] * image[j + 2] as f64;
        buffer[j + 1] += m_y[0][0] * image[j] as f64
            + m_y[0][1] * image[j + 1] as f64
            + m_y[0][2] * image[j + 2] as f64;

        output[j - width + 1] += m_x[1][0] * image[j] as f64
            + m_x[1][1] * image[j + 1] as f64
            + m_x[1][2] * image[j + 2] as f64;
        buffer[j - width + 1] += m_y[1][0] * image[j] as f64
            + m_y[1][1] * image[j + 1] as f64
            + m_y[1][2] * image[j + 2] as f64;
    }
    // (i, j) give start of data block we're applying matrix rows to.
    for i in 2..image.len() / width - 2 {
        for j in 0..width - 2 {
            let data_start = i * width;
            let data = &image[data_start..];
            // Apply each row of kernel to this data chunk and save in appropriate image row.
            output[data_start + j + 1] += m_x[0][0] * data[j] as f64
                + m_x[0][1] * data[j + 1] as f64
                + m_x[0][2] * data[j + 2] as f64;
            buffer[data_start + j + 1] += m_y[0][0] * data[j] as f64
                + m_y[0][1] * data[j + 1] as f64
                + m_y[0][2] * data[j + 2] as f64;

            output[data_start - width + j + 1] += m_x[1][0] * data[j] as f64
                + m_x[1][1] * data[j + 1] as f64
                + m_x[1][2] * data[j + 2] as f64;
            buffer[data_start - width + j + 1] += m_y[1][0] * data[j] as f64
                + m_y[1][1] * data[j + 1] as f64
                + m_y[1][2] * data[j + 2] as f64;

            output[data_start - 2 * width + j + 1] += m_x[2][0] * data[j] as f64
                + m_x[2][1] * data[j + 1] as f64
                + m_x[2][2] * data[j + 2] as f64;
            buffer[data_start - 2 * width + j + 1] += m_y[2][0] * data[j] as f64
                + m_y[2][1] * data[j + 1] as f64
                + m_y[2][2] * data[j + 2] as f64;
        }
    }
    // Bottom rows of input only apply to some rows of output.
    for j in image.len() - 2 * width..image.len() - width - 2 {
        output[j - width + 1] += m_x[1][0] * image[j] as f64
            + m_x[1][1] * image[j + 1] as f64
            + m_x[1][2] * image[j + 2] as f64;
        buffer[j - width + 1] += m_y[1][0] * image[j] as f64
            + m_y[1][1] * image[j + 1] as f64
            + m_y[1][2] * image[j + 2] as f64;

        output[j - 2 * width + 1] += m_x[2][0] * image[j] as f64
            + m_x[2][1] * image[j + 1] as f64
            + m_x[2][2] * image[j + 2] as f64;
        buffer[j - 2 * width + 1] += m_y[2][0] * image[j] as f64
            + m_y[2][1] * image[j + 1] as f64
            + m_y[2][2] * image[j + 2] as f64;
    }
    for j in image.len() - width..image.len() - 2 {
        output[j - 2 * width + 1] += m_x[2][0] * image[j] as f64
            + m_x[2][1] * image[j + 1] as f64
            + m_x[2][2] * image[j + 2] as f64;
        buffer[j - 2 * width + 1] += m_y[2][0] * image[j] as f64
            + m_y[2][1] * image[j + 1] as f64
            + m_y[2][2] * image[j + 2] as f64;
    }

    // Compute magnitude from image gradient components.
    for (m_x, m_y) in output.iter_mut().zip(buffer.iter()) {
        *m_x = (m_x.powi(2) + m_y.powi(2)).sqrt();
    }
}

/// Attempts to use only linear writes to output buffer.
/// It runs over the data three separate times, but it writes
/// to the output buffers linearly, instead of in a strided way.
///
/// Results: It is just slightly slower than sobel_op_test_1.
#[allow(dead_code)]
fn sobel_op_test_2(width: usize, image: &[u8], output: &mut [f64]) {
    assert!(width.is_multiple_of(CACHE_LINE_SIZE));
    assert!(image.len() / width == output.len() / width + 2);

    let m_x = Kernel3X3::sobel_x().m;
    let m_y = Kernel3X3::sobel_y().m;

    let mut buffer = vec![0f64; output.len()];

    // (i, j) give start of data block we're applying matrix rows to.
    for i in 0..image.len() / width - 2 {
        let data_start = i * width;
        let data = &image[data_start..];
        for j in 0..width - 2 {
            // Apply each row of kernel to this data chunk and save in appropriate image row.
            output[data_start + j + 1] += m_x[0][0] * data[j] as f64
                + m_x[0][1] * data[j + 1] as f64
                + m_x[0][2] * data[j + 2] as f64;
            buffer[data_start + j + 1] += m_y[0][0] * data[j] as f64
                + m_y[0][1] * data[j + 1] as f64
                + m_y[0][2] * data[j + 2] as f64;
        }
    }
    for i in 1..image.len() / width - 1 {
        let data_start = i * width;
        let data = &image[data_start..];
        for j in 0..width - 2 {
            // Apply each row of kernel to this data chunk and save in appropriate image row.
            output[data_start - width + j + 1] += m_x[1][0] * data[j] as f64
                + m_x[1][1] * data[j + 1] as f64
                + m_x[1][2] * data[j + 2] as f64;
            buffer[data_start - width + j + 1] += m_y[1][0] * data[j] as f64
                + m_y[1][1] * data[j + 1] as f64
                + m_y[1][2] * data[j + 2] as f64;
        }
    }
    for i in 2..image.len() / width {
        let data_start = i * width;
        let data = &image[data_start..];
        for j in 0..width - 2 {
            // Apply each row of kernel to this data chunk and save in appropriate image row.
            output[data_start - 2 * width + j + 1] += m_x[2][0] * data[j] as f64
                + m_x[2][1] * data[j + 1] as f64
                + m_x[2][2] * data[j + 2] as f64;
            buffer[data_start - 2 * width + j + 1] += m_y[2][0] * data[j] as f64
                + m_y[2][1] * data[j + 1] as f64
                + m_y[2][2] * data[j + 2] as f64;
        }
    }

    // Compute magnitude from image gradient components.
    for (m_x, m_y) in output.iter_mut().zip(buffer.iter()) {
        *m_x = (m_x.powi(2) + m_y.powi(2)).sqrt();
    }
}

pub fn sobel_test(image_proc: &ImageProcessor) {
    let image = &image_proc.image;
    let image_height = image.height() as usize;
    let image_width = image.width() as usize;
    log::info!("Image dimensions (h x w): ({image_height}, {image_width})");

    let start = Instant::now();
    let input_bytes = grayscale_bytes(image);
    log::info!("Time to convert image to grayscale: {:?}.", start.elapsed());

    const DESIRED_THREADS: usize = 16;
    let thread_pool = &image_proc.thread_pool;
    let num_threads = thread_pool.current_num_threads().min(DESIRED_THREADS);

    let start = Instant::now();
    let padded_width = (image_width + 2).next_multiple_of(CACHE_LINE_SIZE);
    let padded_height = image_height.next_multiple_of(num_threads) + 2;
    let mut working_bytes = vec![0u8; padded_height * padded_width];
    for i in 1..image_height {
        for j in 1..image_width {
            working_bytes[i * padded_width + j] = input_bytes[(i - 1) * image_width + (j - 1)];
        }
    }
    let mut output_bytes = vec![0f64; working_bytes.len()];
    log::info!(
        "Time to copy image to padded buffer: {:?}.",
        start.elapsed()
    );
    log::info!(
        "Padded data row width in bytes: input: {} / output: {}",
        (padded_width * std::mem::size_of_val(&working_bytes[0])).to_formatted_string(&Locale::en),
        (padded_width * std::mem::size_of_val(&output_bytes[0])).to_formatted_string(&Locale::en)
    );

    assert!((output_bytes.len() / padded_width - 2).is_multiple_of(num_threads));
    let image_rows = &mut output_bytes[padded_width..working_bytes.len() - padded_width];

    if true {
        log::info!("Performing parallel convolution with {num_threads} threads...");
        let start = Instant::now();
        let pixels_per_thread = image_rows.len() / num_threads;
        thread_pool.install(|| {
            image_rows
                .par_chunks_mut(pixels_per_thread)
                .enumerate()
                .for_each(|(i, chunk)| {
                    sobel_op_test_1(
                        padded_width,
                        &working_bytes
                            [i * pixels_per_thread..(i + 1) * pixels_per_thread + 2 * padded_width],
                        chunk,
                    );
                });
        });
        let elapsed = start.elapsed();
        log::info!("Time to apply parallel sobel operator: {:?}.", elapsed);
    } else {
        let start = Instant::now();
        sobel_op_test_1(padded_width, &working_bytes, image_rows);
        let elapsed = start.elapsed();
        log::info!("Time to apply sobel operator: {:?}.", elapsed);
    }

    if true {
        // Save to file for debugging.
        ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
            padded_width as u32,
            padded_height as u32,
            output_bytes
                .iter()
                .map(|v| v.abs().clamp(0.0, 255.0) as u8)
                .collect(),
        )
        .unwrap()
        .save_with_format(&image_proc.output_path, image::ImageFormat::Jpeg)
        .unwrap();
    }
}

/// Simple direct implementation of the x-direction convolution for the Sobel operator.
///
/// Assumptions: Assumes the data is padded around the image border with 0's.
///
/// Results: This version is significantly faster.
fn sobel_x_conv_naive(height: usize, width: usize, image: &[u8], output: &mut [u8]) {
    assert!(image.len() == output.len() && image.len() == height * width);
    let m_x = Kernel3X3::sobel_x().m;

    for i in 1..height - 1 {
        for j in 1..width - 1 {
            let out_ind = i * width + j;
            let mut val = 0.0f64;
            #[allow(clippy::needless_range_loop)]
            for p in 0..3 {
                for q in 0..3 {
                    let image_ind = ((i + p) - 1) * width + ((j + q) - 1);
                    val += m_x[p][q] * image[image_ind] as f64;
                }
            }
            output[out_ind] = val.abs().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Doesn't skip boundary terms, as we would not copy them to final image.
///
/// Results: Not significantly faster than `sobel_x_conv_naive`.
fn sobel_x_conv_naive_2(height: usize, width: usize, image: &[u8], output: &mut [u8]) {
    assert!(image.len() == output.len() && image.len() == height * width);
    let m_x = Kernel3X3::sobel_x().m;

    // This also updates the boundary padding with meaningless values.
    #[allow(clippy::needless_range_loop)]
    for i in width + 1..(height - 1) * width - 1 {
        let mut val = 0.0f64;
        for p in 0..3 {
            for q in 0..3 {
                let image_ind = i + (p - 1) * width + (q - 1);
                val += m_x[p][q] * image[image_ind] as f64;
            }
        }
        output[i] = val.abs().clamp(0.0, 255.0) as u8;
    }
}

const CACHE_LINE_SIZE: usize = 64;

/// Approach being tried out towards cache optimized implementation of the x-direction Sobel convolution.
///
/// Assumptions: That image and output are stored row-major, with rows of `width` entries,
/// and that image contains two more rows than output.
fn sobel_x_conv_test_1(width: usize, image: &[u8], output: &mut [f64]) {
    assert!(width.is_multiple_of(CACHE_LINE_SIZE));
    assert!(image.len() / width == output.len() / width + 2);
    let m_x = Kernel3X3::sobel_x().m;

    // Top rows of input only contributes to two output rows.
    for j in 0..width {
        output[j + 1] += m_x[0][0] * image[j] as f64
            + m_x[0][1] * image[j + 1] as f64
            + m_x[0][2] * image[j + 2] as f64;
    }
    for j in width..2 * width {
        output[j + 1] += m_x[0][0] * image[j] as f64
            + m_x[0][1] * image[j + 1] as f64
            + m_x[0][2] * image[j + 2] as f64;
        output[j - width + 1] += m_x[1][0] * image[j] as f64
            + m_x[1][1] * image[j + 1] as f64
            + m_x[1][2] * image[j + 2] as f64;
    }
    // (i, j) give start of data block we're applying matrix rows to.
    for i in 2..image.len() / width - 2 {
        for j in 0..(width / CACHE_LINE_SIZE) - 1 {
            let data_start = i * width + j * 64;
            let data = &image[data_start..];
            // Apply each row of kernel to this data chunk and save in appropriate image row.
            for k in 0..64 {
                output[data_start + k + 1] += m_x[0][0] * data[k] as f64
                    + m_x[0][1] * data[k + 1] as f64
                    + m_x[0][2] * data[k + 2] as f64;
                output[data_start - width + k + 1] += m_x[1][0] * data[k] as f64
                    + m_x[1][1] * data[k + 1] as f64
                    + m_x[1][2] * data[k + 2] as f64;
                output[data_start - 2 * width + k + 1] += m_x[2][0] * data[k] as f64
                    + m_x[2][1] * data[k + 1] as f64
                    + m_x[2][2] * data[k + 2] as f64;
            }
        }
        let data_start = i * width + width - 64;
        for k in data_start..(i + 1) * width - 2 {
            output[k + 1] += m_x[0][0] * image[k] as f64
                + m_x[0][1] * image[k + 1] as f64
                + m_x[0][2] * image[k + 2] as f64;
            output[k - width + 1] += m_x[1][0] * image[k] as f64
                + m_x[1][1] * image[k + 1] as f64
                + m_x[1][2] * image[k + 2] as f64;
            output[k - 2 * width + 1] += m_x[2][0] * image[k] as f64
                + m_x[2][1] * image[k + 1] as f64
                + m_x[2][2] * image[k + 2] as f64;
        }
    }
    // Bottom rows of input only apply to some rows of output.
    for j in image.len() - 2 * width..image.len() - width - 2 {
        output[j - width + 1] += m_x[1][0] * image[j] as f64
            + m_x[1][1] * image[j + 1] as f64
            + m_x[1][2] * image[j + 2] as f64;
        output[j - 2 * width + 1] += m_x[2][0] * image[j] as f64
            + m_x[2][1] * image[j + 1] as f64
            + m_x[2][2] * image[j + 2] as f64;
    }
    for j in image.len() - width..image.len() - 2 {
        output[j - 2 * width + 1] += m_x[2][0] * image[j] as f64
            + m_x[2][1] * image[j + 1] as f64
            + m_x[2][2] * image[j + 2] as f64;
    }
}

#[allow(unused)]
#[derive(Subcommand)]
pub enum XConvMethod {
    Test1,
    Test1Parallel,
    Naive1,
    Naive2,
}

pub fn sobel_x_test(image_proc: &ImageProcessor, method: XConvMethod) {
    let image = &image_proc.image;
    let image_height = image.height() as usize;
    let image_width = image.width() as usize;
    log::info!("Image dimensions: ({image_height}, {image_width})");

    let start = Instant::now();
    let input_bytes = grayscale_bytes(image);
    log::info!("Time to convert image to grayscale: {:?}.", start.elapsed());

    let start = Instant::now();
    let padded_width = (image_width + 2).next_multiple_of(CACHE_LINE_SIZE);
    let padded_height = image_height + 2;
    let mut padded_bytes = vec![0u8; padded_height * padded_width];
    for i in 1..image_height {
        for j in 1..image_width {
            padded_bytes[i * padded_width + j] = input_bytes[(i - 1) * image_width + (j - 1)];
        }
    }
    log::info!(
        "Time to copy image to padded buffer: {:?}.",
        start.elapsed()
    );

    const WRITE_IMAGES: bool = false;
    const REPETITIONS: usize = 20;

    match method {
        XConvMethod::Naive1 => {
            for _ in 0..REPETITIONS {
                let mut output_bytes = vec![0u8; padded_bytes.len()];
                let start = Instant::now();
                sobel_x_conv_naive(
                    padded_height,
                    padded_width,
                    &padded_bytes,
                    &mut output_bytes,
                );
                log::info!("Time to apply naive_sobel_x: {:?}.", start.elapsed());

                if WRITE_IMAGES {
                    // Save to file for debugging.
                    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
                        padded_width as u32,
                        padded_height as u32,
                        output_bytes,
                    )
                    .unwrap()
                    .save_with_format("scratch/naive_sobel_test.jpg", image::ImageFormat::Jpeg)
                    .unwrap();
                }
            }
        }

        XConvMethod::Naive2 => {
            for _ in 0..REPETITIONS {
                let mut output_bytes = vec![0u8; padded_bytes.len()];
                let start = Instant::now();
                sobel_x_conv_naive_2(
                    padded_height,
                    padded_width,
                    &padded_bytes,
                    &mut output_bytes,
                );
                log::info!("Time to apply naive_sobel_x_2: {:?}.", start.elapsed());

                if WRITE_IMAGES {
                    // Save to file for debugging.
                    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
                        padded_width as u32,
                        padded_height as u32,
                        output_bytes,
                    )
                    .unwrap()
                    .save_with_format("scratch/naive_sobel_test_2.jpg", image::ImageFormat::Jpeg)
                    .unwrap();
                }
            }
        }

        XConvMethod::Test1 => {
            for _ in 0..REPETITIONS {
                let mut output_bytes = vec![0f64; padded_bytes.len()];
                let start = Instant::now();
                sobel_x_conv_test_1(
                    padded_width,
                    &padded_bytes,
                    // In parallel verson we'll take slice of output as last arg and
                    // slice of input with one row before and one row after same rows
                    // of output for second arg.
                    &mut output_bytes[padded_width..padded_bytes.len() - padded_width],
                );
                log::info!("Time to apply sobel_x_conv_test: {:?}.", start.elapsed());

                if WRITE_IMAGES {
                    // Save to file for debugging.
                    ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
                        padded_width as u32,
                        padded_height as u32,
                        output_bytes
                            .iter()
                            .map(|v| v.abs().clamp(0.0, 255.0) as u8)
                            .collect(),
                    )
                    .unwrap()
                    .save_with_format("scratch/optimized_sobel_test.jpg", image::ImageFormat::Jpeg)
                    .unwrap();
                }
            }
        }

        XConvMethod::Test1Parallel => {
            let thread_pool = &image_proc.thread_pool;
            let num_threads = thread_pool.current_num_threads().min(16);
            let mut output_bytes = vec![0f64; padded_bytes.len()];
            assert!((output_bytes.len() / padded_width - 2).is_multiple_of(num_threads));
            log::info!("Performing parallel convolution with {num_threads} threads...");

            let start = Instant::now();
            let image_rows = &mut output_bytes[padded_width..padded_bytes.len() - padded_width];
            let pixels_per_thread = image_rows.len() / num_threads;
            thread_pool.install(|| {
                image_rows
                    .par_chunks_mut(pixels_per_thread)
                    .enumerate()
                    .for_each(|(i, chunk)| {
                        sobel_x_conv_test_1(
                            padded_width,
                            &padded_bytes[i * pixels_per_thread
                                ..(i + 1) * pixels_per_thread + 2 * padded_width],
                            chunk,
                        );
                    });
            });
            log::info!(
                "Time to apply parallel sobel_x_conv_test: {:?}.",
                start.elapsed()
            );

            if WRITE_IMAGES {
                // Save to file for debugging.
                ImageBuffer::<Luma<u8>, Vec<u8>>::from_vec(
                    padded_width as u32,
                    padded_height as u32,
                    output_bytes
                        .iter()
                        .map(|v| v.abs().clamp(0.0, 255.0) as u8)
                        .collect(),
                )
                .unwrap()
                .save_with_format(
                    "scratch/parallel_optimized_sobel_test.jpg",
                    image::ImageFormat::Jpeg,
                )
                .unwrap();
            }
        }
    }
}

// -------------------------------------
// Basic 3x3 convolution implementation.

pub fn convolve_3x3(image: &ImageProcessor, kernel: Kernel3X3) {
    let width = image.dimensions.0;
    let height = image.dimensions.1;

    let mut out_image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
        width,
        height,
        vec![255; width as usize * height as usize * 4],
    )
    .unwrap();

    let buf = &image.image;
    let m = &kernel.m;

    // 3x3 convolution written out by hand, w/ abs. val. and clamp
    for i in 1..(width - 1) {
        for j in 1..(height - 1) {
            for chan in 0..3 {
                out_image[(i, j)][chan] = (m[0][0] * buf[(i + 1, j + 1)][chan] as f64
                    + m[0][1] * buf[(i + 1, j)][chan] as f64
                    + m[0][2] * buf[(i + 1, j - 1)][chan] as f64
                    + m[1][0] * buf[(i, j + 1)][chan] as f64
                    + m[1][1] * buf[(i, j)][chan] as f64
                    + m[1][2] * buf[(i, j - 1)][chan] as f64
                    + m[2][0] * buf[(i - 1, j + 1)][chan] as f64
                    + m[2][1] * buf[(i - 1, j)][chan] as f64
                    + m[2][2] * buf[(i - 1, j - 1)][chan] as f64)
                    .abs()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }

    // do border convolution

    // clamp coordinates at edges, effectively constant continuation
    let cl = |x: u32, y: u32| (x.clamp(0, width - 1), y.clamp(0, height - 1));

    let conv = |i: u32, j: u32, chan: usize| {
        (m[0][0] * buf[cl(i + 1, j + 1)][chan] as f64
            + m[0][1] * buf[cl(i + 1, j)][chan] as f64
            + m[0][2] * buf[cl(i + 1, j - 1)][chan] as f64
            + m[1][0] * buf[cl(i, j + 1)][chan] as f64
            + m[1][1] * buf[cl(i, j)][chan] as f64
            + m[1][2] * buf[cl(i, j - 1)][chan] as f64
            + m[2][0] * buf[cl(i - 1, j + 1)][chan] as f64
            + m[2][1] * buf[cl(i - 1, j)][chan] as f64
            + m[2][2] * buf[cl(i - 1, j - 1)][chan] as f64)
            .abs()
            .clamp(0.0, 255.0) as u8
    };

    for y in [0, height - 1] {
        for x in 0..width {
            for chan in 0..3 {
                out_image[(x, y)][chan] = conv(x, y, chan);
            }
        }
    }

    for x in [0, width - 1] {
        for y in 0..height {
            for chan in 0..3 {
                out_image[(x, y)][chan] = conv(x, y, chan);
            }
        }
    }

    const OUT_IMAGE_PATH: &str = "test_data/convolution.jpeg";

    out_image
        .save_with_format(OUT_IMAGE_PATH, image::ImageFormat::Jpeg)
        .unwrap();
}

// ----------------------------------
// Current main Sobel implementation.

/// Image data stored row major, in blocks of channels bytes.
pub struct ImageMatrix {
    m: Vec<u8>,
    width: usize,
    #[allow(unused)]
    height: usize,
    channels: usize,
}

impl ImageMatrix {
    pub fn new(width: usize, height: usize, channels: usize, initializer: u8) -> Self {
        Self {
            // could pad rows for cache alignment if useful
            m: vec![initializer; width * height * channels],
            width,
            height,
            channels,
        }
    }

    pub fn to_vec(self) -> Vec<u8> {
        self.m
    }
}

/// indexed by row, column, channel
impl Index<(usize, usize, usize)> for ImageMatrix {
    type Output = u8;
    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.m[self.channels * (index.0 * self.width + index.1) + index.2]
    }
}

impl IndexMut<(usize, usize, usize)> for ImageMatrix {
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.m[self.channels * (index.0 * self.width + index.1) + index.2]
    }
}

fn report_elapsed(time: time::Instant) {
    let elapsed = time.elapsed().as_secs_f32();
    println!("... {:>2.3}s", elapsed);
}

/// Sobel image operator, thresholding coefficients with values below `threshold` to 0.
pub fn sobel(image_proc: &ImageProcessor, threshold: u8) {
    let thread_pool = &image_proc.thread_pool;
    let width = image_proc.dimensions.0;
    let height = image_proc.dimensions.1;

    let buf = &image_proc.image;

    let kernel_x = Kernel3X3::sobel_x();
    let kernel_y = Kernel3X3::sobel_y();

    let m_x = &kernel_x.m;
    let m_y = &kernel_y.m;

    const NUM_CHANNELS: usize = 4;
    const INITIAL_VALUE: u8 = 255;

    let mut matrix = ImageMatrix::new(width as usize, height as usize, NUM_CHANNELS, INITIAL_VALUE);

    // parallelize sobel convolution loop

    let time = time::Instant::now();
    println!("Starting sobel operation.");

    let row_size = width * NUM_CHANNELS as u32;
    let chunk_size_rows = height as usize / thread_pool.current_num_threads();
    let chunk_size = chunk_size_rows * row_size as usize;

    let m_len = matrix.m.len();
    let num_chunks = m_len.div_ceil(chunk_size);

    let flat_index = |index: (usize, usize, usize)| {
        NUM_CHANNELS * (index.0 * width as usize + index.1) + index.2
    };

    let thresh = |v: u8| {
        if v <= threshold { 0 } else { v }
    };

    matrix
        .m
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(i, chunk)| {
            let slice_begin_row = if i == 0 { 1_u32 } else { 0_u32 };

            let slice_end_row = if i == num_chunks - 1 {
                chunk.len() as u32 / row_size - 1
            } else {
                chunk_size_rows as u32
            };

            let y_off_im = (i * chunk_size_rows) as u32;

            // convolution operations written out by hand
            for y in slice_begin_row..slice_end_row {
                for x in 1..(width - 1) {
                    for chan in 0..3 {
                        let o_x = m_x[0][0] * buf[(x + 1, y_off_im + y + 1)][chan] as f64
                            + m_x[0][1] * buf[(x + 1, y_off_im + y)][chan] as f64
                            + m_x[0][2] * buf[(x + 1, y_off_im + y - 1)][chan] as f64
                            + m_x[1][0] * buf[(x, y_off_im + y + 1)][chan] as f64
                            + m_x[1][1] * buf[(x, y_off_im + y)][chan] as f64
                            + m_x[1][2] * buf[(x, y_off_im + y - 1)][chan] as f64
                            + m_x[2][0] * buf[(x - 1, y_off_im + y + 1)][chan] as f64
                            + m_x[2][1] * buf[(x - 1, y_off_im + y)][chan] as f64
                            + m_x[2][2] * buf[(x - 1, y_off_im + y - 1)][chan] as f64;

                        let o_y = m_y[0][0] * buf[(x + 1, y_off_im + y + 1)][chan] as f64
                            + m_y[0][1] * buf[(x + 1, y_off_im + y)][chan] as f64
                            + m_y[0][2] * buf[(x + 1, y_off_im + y - 1)][chan] as f64
                            + m_y[1][0] * buf[(x, y_off_im + y + 1)][chan] as f64
                            + m_y[1][1] * buf[(x, y_off_im + y)][chan] as f64
                            + m_y[1][2] * buf[(x, y_off_im + y - 1)][chan] as f64
                            + m_y[2][0] * buf[(x - 1, y_off_im + y + 1)][chan] as f64
                            + m_y[2][1] * buf[(x - 1, y_off_im + y)][chan] as f64
                            + m_y[2][2] * buf[(x - 1, y_off_im + y - 1)][chan] as f64;

                        // matrix index is (row, column) vs. (x, y) for image
                        chunk[flat_index((y as usize, x as usize, chan))] =
                            thresh((o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8);
                    }
                }
            }
        });

    report_elapsed(time);

    // do border convolutions

    // clamp coordinates at edges, effectively constant continuation
    let clamp = |x: u32, y: u32| (x.clamp(0, width - 1), y.clamp(0, height - 1));

    let directional_derivs = |x: u32, y: u32, chan: usize| {
        let o_x = m_x[0][0] * buf[clamp(x + 1, y + 1)][chan] as f64
            + m_x[0][1] * buf[clamp(x + 1, y)][chan] as f64
            + m_x[0][2] * buf[clamp(x + 1, y - 1)][chan] as f64
            + m_x[1][0] * buf[clamp(x, y + 1)][chan] as f64
            + m_x[1][1] * buf[clamp(x, y)][chan] as f64
            + m_x[1][2] * buf[clamp(x, y - 1)][chan] as f64
            + m_x[2][0] * buf[clamp(x - 1, y + 1)][chan] as f64
            + m_x[2][1] * buf[clamp(x - 1, y)][chan] as f64
            + m_x[2][2] * buf[clamp(x - 1, y - 1)][chan] as f64;

        let o_y = m_y[0][0] * buf[clamp(x + 1, y + 1)][chan] as f64
            + m_y[0][1] * buf[clamp(x + 1, y)][chan] as f64
            + m_y[0][2] * buf[clamp(x + 1, y - 1)][chan] as f64
            + m_y[1][0] * buf[clamp(x, y + 1)][chan] as f64
            + m_y[1][1] * buf[clamp(x, y)][chan] as f64
            + m_y[1][2] * buf[clamp(x, y - 1)][chan] as f64
            + m_y[2][0] * buf[clamp(x - 1, y + 1)][chan] as f64
            + m_y[2][1] * buf[clamp(x - 1, y)][chan] as f64
            + m_y[2][2] * buf[clamp(x - 1, y - 1)][chan] as f64;

        (o_x, o_y)
    };

    for y in [0, height - 1] {
        for x in 0..width {
            for chan in 0..3 {
                let (o_x, o_y) = directional_derivs(x, y, chan);
                matrix[(y as usize, x as usize, chan)] =
                    thresh((o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8);
            }
        }
    }

    for x in [0, width - 1] {
        for y in 0..height {
            for chan in 0..3 {
                let (o_x, o_y) = directional_derivs(x, y, chan);
                matrix[(y as usize, x as usize, chan)] =
                    thresh((o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8);
            }
        }
    }

    let out_image =
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, matrix.to_vec()).unwrap();

    let time = time::Instant::now();
    println!("Writing image to disk.");

    out_image
        .save_with_format(&image_proc.output_path, image::ImageFormat::Jpeg)
        .unwrap();

    report_elapsed(time);
}
