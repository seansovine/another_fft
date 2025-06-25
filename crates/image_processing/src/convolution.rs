// Code for implementing convolution filters on images.
//
// This version is not optimized.

use crate::Image;
use image::{ImageBuffer, Rgba};

#[allow(unused)]
trait SizedKernel {
    const WIDTH: usize;
    const HEIGHT: usize;
    const CENTER: (usize, usize);
}

pub struct Kernel3X3 {
    m: [[f32; 3]; 3],
}

impl SizedKernel for Kernel3X3 {
    const WIDTH: usize = 3;
    const HEIGHT: usize = 3;
    const CENTER: (usize, usize) = (1, 1);
}

impl Kernel3X3 {
    pub fn sobel_x() -> Self {
        Self {
            m: [
                [-1.0, 0.0, 1.0], //
                [-2.0, 0.0, 2.0], //
                [-1.0, 0.0, 1.0], //
            ],
        }
    }

    pub fn sobel_y() -> Self {
        Self {
            m: [
                [-1.0, -2.0, -1.0], //
                [0.0, 0.0, 0.0],    //
                [1.0, 2.0, 1.0],    //
            ],
        }
    }
}

pub fn convolve_3x3(image: &Image, kernel: Kernel3X3) {
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

    // 3x3 convolution written out by hand, w/ abs. value
    for i in 1..(width - 1) {
        for j in 1..(height - 1) {
            for chan in 0..3 {
                out_image[(i, j)][chan] = (
                    //
                    m[0][0] * buf[(i + 1, j + 1)][chan] as f32
                        + m[0][1] * buf[(i + 1, j)][chan] as f32
                        + m[0][2] * buf[(i + 1, j - 1)][chan] as f32
                        + m[1][0] * buf[(i, j + 1)][chan] as f32
                        + m[1][1] * buf[(i, j)][chan] as f32
                        + m[1][2] * buf[(i, j - 1)][chan] as f32
                        + m[2][0] * buf[(i - 1, j + 1)][chan] as f32
                        + m[2][1] * buf[(i - 1, j)][chan] as f32
                        + m[2][2] * buf[(i - 1, j - 1)][chan] as f32
                    //
                )
                    .abs()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }

    // TODO: implement outside border convolution

    const OUT_IMAGE_PATH: &str = "test_data/convolution.jpeg";

    out_image
        .save_with_format(OUT_IMAGE_PATH, image::ImageFormat::Jpeg)
        .unwrap();
}

pub fn sobel(image: &Image) {
    let width = image.dimensions.0;
    let height = image.dimensions.1;

    let mut out_image = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(
        width,
        height,
        vec![255; width as usize * height as usize * 4],
    )
    .unwrap();

    let buf = &image.image;

    let kernel_x = Kernel3X3::sobel_x();
    let kernel_y = Kernel3X3::sobel_y();

    let m_x = &kernel_x.m;
    let m_y = &kernel_y.m;

    // sobel convolution written out by hand
    for i in 1..(width - 1) {
        for j in 1..(height - 1) {
            for chan in 0..3 {
                let o_x = m_x[0][0] * buf[(i + 1, j + 1)][chan] as f32
                    + m_x[0][1] * buf[(i + 1, j)][chan] as f32
                    + m_x[0][2] * buf[(i + 1, j - 1)][chan] as f32
                    + m_x[1][0] * buf[(i, j + 1)][chan] as f32
                    + m_x[1][1] * buf[(i, j)][chan] as f32
                    + m_x[1][2] * buf[(i, j - 1)][chan] as f32
                    + m_x[2][0] * buf[(i - 1, j + 1)][chan] as f32
                    + m_x[2][1] * buf[(i - 1, j)][chan] as f32
                    + m_x[2][2] * buf[(i - 1, j - 1)][chan] as f32;

                let o_y = m_y[0][0] * buf[(i + 1, j + 1)][chan] as f32
                    + m_y[0][1] * buf[(i + 1, j)][chan] as f32
                    + m_y[0][2] * buf[(i + 1, j - 1)][chan] as f32
                    + m_y[1][0] * buf[(i, j + 1)][chan] as f32
                    + m_y[1][1] * buf[(i, j)][chan] as f32
                    + m_y[1][2] * buf[(i, j - 1)][chan] as f32
                    + m_y[2][0] * buf[(i - 1, j + 1)][chan] as f32
                    + m_y[2][1] * buf[(i - 1, j)][chan] as f32
                    + m_y[2][2] * buf[(i - 1, j - 1)][chan] as f32;

                out_image[(i, j)][chan] =
                    (o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8;
            }
        }
    }

    // TODO: implement outside border convolution

    const OUT_IMAGE_PATH: &str = "test_data/sobel.jpeg";

    out_image
        .save_with_format(OUT_IMAGE_PATH, image::ImageFormat::Jpeg)
        .unwrap();
}
