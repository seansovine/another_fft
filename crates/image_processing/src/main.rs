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
}

#[derive(Debug, Args)]
pub struct ResizeArgs {
  #[clap(long, required = true)]
  path: String,
}

fn main() {
  let args = CliArgs::parse();

  match args.command {
    Command::Test => test(),
    Command::Resize(args) => image_processing::resize::resize(&args.path),
  }
}

fn test() {
  println!("Running FFT test:");
  image_processing::fft_test();
}
