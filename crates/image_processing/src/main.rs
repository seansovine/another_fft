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
    let args = CliArgs::parse();
    let path = &args.path;
    let image_processor = image_processing::ImageProcessor::from_path(path)?;

    match args.command {
        Command::Resize(args) => {
            image_processing::basic_ops::resize(&image_processor, args.new_width, args.new_height)?;
        }
        Command::Grayscale => {
            image_processing::basic_ops::to_grayscale(image_processor)?;
        }
        Command::Fft(args) => {
            image_processing::fft::fft_image(&image_processor, args.filter)?;
        }
        Command::Convolve => {
            // choose this as an example; add arg later
            let kernel = image_processing::convolution::Kernel3X3::sobel_y();
            image_processing::convolution::convolve_3x3(&image_processor, kernel);
        }
        Command::Sobel => {
            image_processing::convolution::sobel(&image_processor);
        }
    }

    Ok(())
}
