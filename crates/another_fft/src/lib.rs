#[cfg(test)]
mod tests;

// Home of functions to compute the FFT.

use rayon::{ThreadPool, prelude::*};
use std::f64::consts::PI;

/// This is the implementation from
///
/// ```text
/// Press, et al., _Numerical Reciples_, Third Edition.
/// Cambridge University Press, 2007.
/// ```
///
/// which the authors cite as originating in the work of
/// N.M. Brenner. (See Rader and Brenner, 1976.)
///
/// __Arguments:__
///
/// + `data` - the sequence of N complex values with real parts in
///   even-numbered entries and imaginary parts in odd entries.
///
/// + `inverse` - if true will compute the inverse transform;
///   the convention here is to put the 1/N normalizing factor
///   on the inverse transform.
///
pub fn fft(data: &mut [f64], inverse: bool) {
    let data_len = data.len() / 2;
    if (data_len as f64).log2().fract() != 0.0 {
        panic!("FFT routine is only implemented for arrays with length a power of 2.")
    }

    let n = data.len() / 2;
    let nn = n << 1;

    // phase sign and normalizing factor for inverse transform
    let inv_sign: f64 = if inverse { 1f64 } else { -1f64 };
    let inv_mult = if inverse { 1f64 / (n as f64) } else { 1f64 };

    // Bit reversal portion:
    //
    //  This algorithm swaps each complex data entry (pair of slice entries)
    //  with the entry whose index has the binary representation that is the
    //  reverse of its binary representation.

    let mut j: usize = 1;
    let mut m: usize;

    for i in (1..nn).step_by(2) {
        if j > i {
            data.swap(j - 1, i - 1);
            data.swap(j, i);
        }

        // compute next potential swap row
        m = n;
        while m >= 2 && j > m {
            j -= m;
            m >>= 1;
        }
        j += m;

        // divide by complex sequence length if inverse
        data[i - 1] *= inv_mult;
        data[i] *= inv_mult;
    }

    // Danielson-Lanczos iteration:
    //
    //  The next part performs the "butterfly" operations on the vector
    //  that was resorted in the previous step. A recurrence is used to
    //  efficiently compute the trigonmetric functions involved.

    let mut mmax: usize = 2;
    let mut istep: usize;

    let mut theta: f64;
    let mut wtemp: f64;
    let mut wpr: f64;
    let mut wpi: f64;

    while nn > mmax {
        istep = mmax << 1;

        theta = inv_sign * 2f64 * PI / (mmax as f64);
        wtemp = (theta / 2f64).sin();

        wpr = -2.0f64 * wtemp * wtemp;
        wpi = theta.sin();

        // start recurrence for trigonometric functions of whole multiples of a given angle
        let mut wr: f64 = 1.0;
        let mut wi: f64 = 0.0;

        let mut tempr: f64;
        let mut tempi: f64;

        for m in (1..mmax).step_by(2) {
            for i in (m..=nn).step_by(istep) {
                j = i + mmax;
                tempr = wr * data[j - 1] - wi * data[j];
                tempi = wr * data[j] + wi * data[j - 1];

                data[j - 1] = data[i - 1] - tempr;
                data[j] = data[i] - tempi;
                data[i - 1] += tempr;
                data[i] += tempi;
            }

            wtemp = wr;
            wr = wr * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }

        mmax = istep;
    }
}

// 2D FFT implementation

/// Basic implementation that works by applying the row-wise fft, transposing
/// and applying the row-wise fft again, and then transposing again.
///
/// This version allocates a temporary array as part of the transposition, but
/// a clever in-place algorithm for the transposition of a flattened array may
/// exist.
///
/// __Arguments:__
///
/// + `data` - flattened 2D array of (real, complex) pairs
///
/// + `dimensions` - (width, height) of array of _complex_ values
///
/// + `inverse` - if true will compute the inverse transform;
///   the convention here is to put the 1/MN normalizing factor
///   on the inverse transform.
///
pub fn fft_2d(data: &mut [f64], dimensions: (usize, usize), inverse: bool) {
    // width in terms of conceptual complex data
    // entries; data width is twice this number
    let width = dimensions.0;
    let height = dimensions.1;

    // row-wise fft
    for i in 0..height {
        let offset = i * width * 2;
        fft(&mut data[offset..offset + width * 2], inverse);
    }

    // we just allocate an intermediate array; a clever in-place algorithm for
    // transposing a matrix in flattened form may exist, but this works for now
    let mut transposed: Vec<f64> = vec![0.0_f64; 2 * width * height];

    complex_transpose(data, dimensions, &mut transposed);
    // dimensions are now reversed

    // column-wise fft (row-wise on transposed matrix)
    for i in 0..width {
        let offset = i * height * 2;
        fft(&mut transposed[offset..offset + height * 2], inverse);
    }

    // transpose back now
    complex_transpose(&transposed, (dimensions.1, dimensions.0), data);
}

/// Transpose a matrix of complex entries represented as double-width
/// matrix of real entries.
///
/// __Note:__
///
/// There is possibly a clever way to do the transposed on the flattened
/// array in-place, but here we copy into a separate buffor for simplicity.
///
fn complex_transpose(data_in: &[f64], in_dimensions: (usize, usize), data_out: &mut [f64]) {
    let width = in_dimensions.0;
    let height = in_dimensions.1;

    for i in 0..height {
        for j in 0..width {
            // original and new real-part indices
            let n_r = 2 * (i * width + j);
            let n_r_new = (j * height + i) * 2;

            // swap real parts
            data_out[n_r_new] = data_in[n_r];
            // swap complex parts -
            // source and destination adjacent to real parts
            data_out[n_r_new + 1] = data_in[n_r + 1];
        }
    }
}

// parallel 2d FFT implementation

fn fft_2d_para_internal(
    data: &mut [f64],
    working_buffer: &mut [f64],
    dimensions: (usize, usize),
    inverse: bool,
) {
    assert!(working_buffer.len() == data.len());

    // width in terms of conceptual complex data
    // entries; data width is twice this number
    let width = dimensions.0;
    let height = dimensions.1;

    // perform the row-wise FFT in parallel, letting Rayon handle
    // the details; each "job" fpr rayon is a row of the matrix
    data.par_chunks_exact_mut(width * 2)
        .for_each(|slice| fft(slice, inverse));

    complex_transpose(data, dimensions, working_buffer);
    // dimensions are now reversed

    // each job is a row of transposed matrix
    working_buffer
        .par_chunks_exact_mut(height * 2)
        .for_each(|slice| fft(slice, inverse));

    // transpose back now
    complex_transpose(&*working_buffer, (dimensions.1, dimensions.0), data);
}

/// Basic parallel implementation that works by applying the row-wise fft,
/// transposing and applying the row-wise fft again, and then transposing again.
/// The parallelization is done on the computation of row-wise FFTs.
///
/// Note that this implementation takes in a working buffer the same size as
/// the image data, that it uses to hold the transposed image matrix in the
/// intermediate step. The transpose operation is not done in parallel and adds
/// significant time to the computation.
///
/// __Arguments:__
///
/// + `data` - flattened 2D array of (real, complex) pairs
///
/// + `working_buffer` - flattened 2D array of (real, complex) pairs the same size
///   as `data`, for use in intermediate steps of the FFT computation
///
/// + `dimensions` - (width, height) of array of _complex_ values
///
/// + `inverse` - if true will compute the inverse transform;
///   the convention here is to put the 1/MN normalizing factor
///   on the inverse transform.
///
/// + `thread_pool` - Rayon thread pool to execute the computation within
///
pub fn fft_2d_para(
    data: &mut [f64],
    working_buffer: &mut [f64],
    dimensions: (usize, usize),
    inverse: bool,
    thread_pool: &ThreadPool,
) {
    // run parallel fft in rayon thread pool
    thread_pool.install(|| fft_2d_para_internal(data, working_buffer, dimensions, inverse));
}
