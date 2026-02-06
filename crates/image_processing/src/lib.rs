// top-level library module

pub mod basic_ops;
pub mod convolution;
pub mod fft;

use image::{ImageBuffer, Rgba};
use rayon::ThreadPool;

pub struct ImageProcessor {
    // input image data
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    pub dimensions: (u32, u32),

    // Rayon thread pool for parallel image operations
    pub thread_pool: ThreadPool,

    // Path to save result image to.
    pub output_path: String,
}

impl ImageProcessor {
    pub fn from_path(filepath: &str, output_path: &str) -> Result<Self, String> {
        let Ok(image_bytes) = std::fs::read(filepath) else {
            return Err(format!("Unable to read image at path: {}", filepath));
        };

        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba8();
        let dimensions = image.dimensions();

        let num_threads = num_cpus::get();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        Ok(Self {
            image,
            dimensions,
            thread_pool,
            output_path: output_path.into(),
        })
    }
}
