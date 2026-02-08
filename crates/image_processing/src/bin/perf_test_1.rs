//! An example program for analyzing program runs with perf.

use std::time::Instant;

const DATA_SIZE: usize = 7952 * 5304 * 10;

fn main() {
    env_logger::init();
    let data = vec![1u8; DATA_SIZE];
    let mut output = vec![0f64; DATA_SIZE];

    let start = Instant::now();
    for i in 0..data.len() {
        output[i] += data[i] as f64;
    }
    let elapsed = start.elapsed();
    log::info!("Output[0]: {}", output[0]);
    log::info!("Time to add numbers linear: {:?}", elapsed);
}
