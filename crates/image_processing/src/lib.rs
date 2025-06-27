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

    pub thread_pool: ThreadPool,
}

impl ImageProcessor {
    pub fn from_path(filepath: &str) -> Result<Self, String> {
        let Ok(image_bytes) = std::fs::read(filepath) else {
            return Err(format!("Unable to read image at path: {}", filepath));
        };

        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba8();
        let dimensions = image.dimensions();

        #[cfg(feature = "timing")]
        println!("Performing initial setup.");
        #[cfg(feature = "timing")]
        let time = time::Instant::now();

        let num_threads = num_cpus::get();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();

        #[cfg(feature = "timing")]
        println!(
            "... Rayon setup took {:>2.3}s",
            time.elapsed().as_secs_f32()
        );

        Ok(Self {
            image,
            dimensions,
            thread_pool,
        })
    }
}
