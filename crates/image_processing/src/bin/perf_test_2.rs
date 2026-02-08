//! A different program function for analyzing program runs with perf.

use std::time::Instant;

const DATA_SIZE: usize = 7952 * 5304 * 10;

fn main() {
    env_logger::init();
    let data = vec![1u8; DATA_SIZE];
    let mut output = vec![0f64; DATA_SIZE];

    const STRIDES: usize = 128;
    assert!(data.len().is_multiple_of(STRIDES));

    let start = Instant::now();
    for i in 0..data.len() / STRIDES {
        for j in 0..STRIDES {
            output[j * data.len() / STRIDES + i] += data[j * data.len() / STRIDES + i] as f64;
        }
    }
    let elapsted = start.elapsed();
    log::info!("Output[0]: {}", output[0]);
    log::info!("Time to add numbers strided: {:?}", elapsted);
}

// Notes:
//  Perf shows that this version has a much higher rate of L1 cache loads and
//  roughly three times as many last level cache misses going all the way to RAM.
//
//  So it looks like the strided reads and writes cause much more cache work.
//  But the overall runtime of the loop is not that much great. Interesting.
