// CLI for image processing crate that uses another_fft

use clap::{Args, Parser, Subcommand};

// setup command line args

#[derive(Parser)]
pub struct CliArgs {
    #[clap(subcommand)]
    pub command: Command,
    #[clap(long, required = true)]
    path: String,
}

#[derive(Subcommand)]
pub enum Command {
    Resize(ResizeArgs),
    Grayscale,
    Fft(FftArgs),
    Convolve,
    Sobel,
    OptTest,
}

#[derive(Debug, Args)]
pub struct ResizeArgs {
    #[clap(required = true)]
    new_width: usize,
    #[clap(required = true)]
    new_height: usize,
}

#[derive(Debug, Args)]
pub struct FftArgs {
    #[clap(long, action)]
    filter: bool,
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = CliArgs::parse();
    let path = &args.path;
    let image_processor = image_processing::ImageProcessor::from_path(path)?;

    match args.command {
        Command::Resize(args) => {
            image_processing::basic_ops::resize(&image_processor, args.new_width, args.new_height)?;
        }
        Command::Grayscale => {
            image_processing::basic_ops::save_grayscale(image_processor)?;
        }
        Command::Fft(args) => {
            image_processing::fft::fft_image(&image_processor, args.filter)?;
        }
        Command::Convolve => {
            // hard code for now; add arg later
            let kernel = image_processing::convolution::Kernel3X3::avg();
            image_processing::convolution::convolve_3x3(&image_processor, kernel);
        }
        Command::Sobel => {
            // hard code for now; add arg later
            let threshold: u8 = 75;
            image_processing::convolution::sobel(&image_processor, threshold);
        }
        Command::OptTest => {
            image_processing::convolution::sobel_x_optimized(&image_processor);
        }
    }

    Ok(())
}
