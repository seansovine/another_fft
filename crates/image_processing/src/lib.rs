// top-level library module

pub mod basic_ops;
pub mod convolution;
pub mod fft;

use image::{ImageBuffer, Rgba};

// borrowed from wgpu_grapher

pub struct Image {
    pub image: ImageBuffer<Rgba<u8>, Vec<u8>>,
    pub dimensions: (u32, u32),
}

impl Image {
    pub fn from_file(filepath: &str) -> Result<Self, String> {
        let Ok(image_bytes) = std::fs::read(filepath) else {
            return Err(format!("Unable to read image at path: {}", filepath));
        };

        let image = image::load_from_memory(&image_bytes).unwrap().to_rgba8();
        let dimensions = image.dimensions();

        Ok(Self { image, dimensions })
    }
}
