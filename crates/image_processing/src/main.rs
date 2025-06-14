// CLI for image processing crate that uses another_fft

use clap::{Args, Parser, Subcommand};

// setup command line args

#[derive(Parser)]
pub struct CliArgs {
  #[clap(subcommand)]
  pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
  Test,
  Resize(ResizeArgs),
  Grayscale(PathArgs),
  Fft(PathArgs),
}

#[derive(Debug, Args)]
pub struct ResizeArgs {
  #[clap(long, required = true)]
  path: String,
  #[clap(required = true)]
  new_width: usize,
  #[clap(required = true)]
  new_height: usize,
}

#[derive(Debug, Args)]
pub struct PathArgs {
  #[clap(long, required = true)]
  path: String,
}

fn main() {
  let args = CliArgs::parse();

  match args.command {
    Command::Test => test(),
    Command::Resize(args) => image_processing::basic_ops::resize(&args.path, args.new_width, args.new_height),
    Command::Grayscale(args) => image_processing::basic_ops::to_grayscale(&args.path),
    Command::Fft(args) => image_processing::fft::fft_image(&args.path),
  }
}

fn test() {
  println!("Running FFT test:");
  image_processing::fft_test();
}
