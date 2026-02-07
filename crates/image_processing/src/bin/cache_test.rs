//! Start of a program to play around with analyzing cache misses with perf.

use std::time::Instant;

fn main() {
    env_logger::init();

    const DATA_SIZE: usize = 7952 * 5304 * 100;
    let data = vec![1u8; DATA_SIZE];
    let mut sum = 0f64;

    if false {
        let start = Instant::now();
        for num in data {
            sum += num as f64;
        }
        let elapsted = start.elapsed();
        log::info!("Sum: {sum}");
        log::info!("Time to add numbers linear: {:?}", elapsted);
    } else {
        let data = vec![1u8; DATA_SIZE];
        const STRIDES: usize = 16;
        assert!(data.len().is_multiple_of(STRIDES));
        sum = 0f64;

        let start = Instant::now();
        for i in 0..data.len() / STRIDES - 2 {
            for j in 0..STRIDES {
                sum += data[j * data.len() / STRIDES + i] as f64
                    + data[j * data.len() / STRIDES + i + 1] as f64  // or 65
                    + data[j * data.len() / STRIDES + i + 2] as f64; // or 129
            }
        }
        let elapsted = start.elapsed();
        log::info!("Sum: {}", sum);
        log::info!("Time to add numbers strided: {:?}", elapsted);
    }
}

// See stats on cache miss percentage, instruction throughput, etc.:
//  sudo perf stat -B -e cache-references,cache-misses,cycles,instructions,branches,faults,migrations ./target/release/cache_test
// or
//  sudo perf stat -d ./target/release/cache_test
