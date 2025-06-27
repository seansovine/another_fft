// Code for implementing convolution filters on images.
//
// This version is not optimized.

use crate::ImageProcessor;
use image::{ImageBuffer, Rgba};
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};
use std::ops::{Index, IndexMut};
use std::time;

pub struct Kernel3X3 {
    m: [[f32; 3]; 3],
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

#[allow(unused)]
pub struct ImageMatrix {
    // data stored row major
    m: Vec<u8>,

    width: usize,
    height: usize,
    channels: usize,
}

impl ImageMatrix {
    pub fn new(width: usize, height: usize, channels: usize, initializer: u8) -> Self {
        Self {
            // NOTE: we could pad rows to ensure cache alignment
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

impl Index<(usize, usize, usize)> for ImageMatrix {
    type Output = u8;
    // index is (row, column, channel)
    fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
        &self.m[self.channels * (index.0 * self.width + index.1) + index.2]
    }
}

impl IndexMut<(usize, usize, usize)> for ImageMatrix {
    // index is (row, column, channel)
    fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
        &mut self.m[self.channels * (index.0 * self.width + index.1) + index.2]
    }
}

fn report_elapsed(time: time::Instant) {
    let elapsed = time.elapsed().as_secs_f32();
    println!("... {:>2.3}s", elapsed);
}

pub fn sobel(image: &ImageProcessor, threshold: u8) {
    let thread_pool = &image.thread_pool;
    let width = image.dimensions.0;
    let height = image.dimensions.1;

    let buf = &image.image;

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

            for y in slice_begin_row..slice_end_row {
                for x in 1..(width - 1) {
                    for chan in 0..3 {
                        let o_x = m_x[0][0] * buf[(x + 1, y_off_im + y + 1)][chan] as f32
                            + m_x[0][1] * buf[(x + 1, y_off_im + y)][chan] as f32
                            + m_x[0][2] * buf[(x + 1, y_off_im + y - 1)][chan] as f32
                            + m_x[1][0] * buf[(x, y_off_im + y + 1)][chan] as f32
                            + m_x[1][1] * buf[(x, y_off_im + y)][chan] as f32
                            + m_x[1][2] * buf[(x, y_off_im + y - 1)][chan] as f32
                            + m_x[2][0] * buf[(x - 1, y_off_im + y + 1)][chan] as f32
                            + m_x[2][1] * buf[(x - 1, y_off_im + y)][chan] as f32
                            + m_x[2][2] * buf[(x - 1, y_off_im + y - 1)][chan] as f32;

                        let o_y = m_y[0][0] * buf[(x + 1, y_off_im + y + 1)][chan] as f32
                            + m_y[0][1] * buf[(x + 1, y_off_im + y)][chan] as f32
                            + m_y[0][2] * buf[(x + 1, y_off_im + y - 1)][chan] as f32
                            + m_y[1][0] * buf[(x, y_off_im + y + 1)][chan] as f32
                            + m_y[1][1] * buf[(x, y_off_im + y)][chan] as f32
                            + m_y[1][2] * buf[(x, y_off_im + y - 1)][chan] as f32
                            + m_y[2][0] * buf[(x - 1, y_off_im + y + 1)][chan] as f32
                            + m_y[2][1] * buf[(x - 1, y_off_im + y)][chan] as f32
                            + m_y[2][2] * buf[(x - 1, y_off_im + y - 1)][chan] as f32;

                        // note: matrix index is (row, column) vs. (x, y)
                        chunk[flat_index((y as usize, x as usize, chan))] =
                            thresh((o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8);
                    }
                }
            }
        });

    report_elapsed(time);

    // do border convolutions

    // we act as if the image were constant outside the border
    let clamp = |x: u32, y: u32| (x.clamp(0, width - 1), y.clamp(0, height - 1));

    let directional_derivs = |x: u32, y: u32, chan: usize| {
        let o_x = m_x[0][0] * buf[clamp(x + 1, y + 1)][chan] as f32
            + m_x[0][1] * buf[clamp(x + 1, y)][chan] as f32
            + m_x[0][2] * buf[clamp(x + 1, y - 1)][chan] as f32
            + m_x[1][0] * buf[clamp(x, y + 1)][chan] as f32
            + m_x[1][1] * buf[clamp(x, y)][chan] as f32
            + m_x[1][2] * buf[clamp(x, y - 1)][chan] as f32
            + m_x[2][0] * buf[clamp(x - 1, y + 1)][chan] as f32
            + m_x[2][1] * buf[clamp(x - 1, y)][chan] as f32
            + m_x[2][2] * buf[clamp(x - 1, y - 1)][chan] as f32;

        let o_y = m_y[0][0] * buf[clamp(x + 1, y + 1)][chan] as f32
            + m_y[0][1] * buf[clamp(x + 1, y)][chan] as f32
            + m_y[0][2] * buf[clamp(x + 1, y - 1)][chan] as f32
            + m_y[1][0] * buf[clamp(x, y + 1)][chan] as f32
            + m_y[1][1] * buf[clamp(x, y)][chan] as f32
            + m_y[1][2] * buf[clamp(x, y - 1)][chan] as f32
            + m_y[2][0] * buf[clamp(x - 1, y + 1)][chan] as f32
            + m_y[2][1] * buf[clamp(x - 1, y)][chan] as f32
            + m_y[2][2] * buf[clamp(x - 1, y - 1)][chan] as f32;

        (o_x, o_y)
    };

    for y in [0, height - 1] {
        for x in 0..width {
            for chan in 0..3 {
                let (o_x, o_y) = directional_derivs(x, y, chan);

                // note: matrix index is (row, column) vs. (x, y)
                matrix[(y as usize, x as usize, chan)] =
                    thresh((o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8);
            }
        }
    }

    for x in [0, width - 1] {
        for y in 0..height {
            for chan in 0..3 {
                let (o_x, o_y) = directional_derivs(x, y, chan);

                // note: matrix index is (row, column) vs. (x, y)
                matrix[(y as usize, x as usize, chan)] =
                    thresh((o_x.powi(2) + o_y.powi(2)).sqrt().clamp(0.0, 255.0) as u8);
            }
        }
    }

    let out_image =
        ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(width, height, matrix.to_vec()).unwrap();

    const OUT_IMAGE_PATH: &str = "test_data/sobel.jpeg";

    let time = time::Instant::now();
    println!("Writing image to disk.");

    out_image
        .save_with_format(OUT_IMAGE_PATH, image::ImageFormat::Jpeg)
        .unwrap();

    report_elapsed(time);
}
