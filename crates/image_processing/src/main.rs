// CLI for image processing crate that uses another_fft

use clap::{Args, Parser, Subcommand};
use image_processing::convolution::XConvMethod;

// setup command line args

#[derive(Parser)]
pub struct CliArgs {
    #[clap(subcommand)]
    pub command: Command,
    #[clap(long, required = true)]
    path: String,
    #[clap(long, required = true)]
    output_path: String,
}

#[derive(Subcommand)]
pub enum Command {
    Resize(ResizeArgs),
    Grayscale,
    Fft(FftArgs),
    Convolve,
    Sobel,
    Sobel2,
    ConvTest(ConvTestArgs),
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

#[derive(Args)]
pub struct ConvTestArgs {
    #[clap(subcommand)]
    method: XConvMethod,
}

fn main() -> Result<(), String> {
    env_logger::init();
    let args = CliArgs::parse();
    let image_processor =
        image_processing::ImageProcessor::from_path(&args.path, &args.output_path)?;

    match args.command {
        Command::Resize(args) => {
            image_processing::basic_ops::save_resize(
                &image_processor,
                args.new_width,
                args.new_height,
            )?;
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
            let threshold: u8 = 0;
            image_processing::convolution::sobel(&image_processor, threshold);
        }
        Command::Sobel2 => {
            image_processing::convolution::sobel_test(&image_processor);
        }
        Command::ConvTest(args) => {
            image_processing::convolution::sobel_x_test(&image_processor, args.method);
        }
    }

    Ok(())
}
