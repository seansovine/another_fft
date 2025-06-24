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

    match args.command {
        Command::Resize(args) => {
            image_processing::basic_ops::resize(path, args.new_width, args.new_height)?
        }
        Command::Grayscale => image_processing::basic_ops::to_grayscale(&args.path)?,
        Command::Fft(args) => image_processing::fft::fft_image(path, args.filter)?,
    }

    Ok(())
}
