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
  Resize(PathArgs),
  Grayscale(PathArgs),
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
    Command::Resize(args) => image_processing::basic_ops::resize(&args.path),
    Command::Grayscale(args) => image_processing::basic_ops::to_grayscale(&args.path),
  }
}

fn test() {
  println!("Running FFT test:");
  image_processing::fft_test();
}
