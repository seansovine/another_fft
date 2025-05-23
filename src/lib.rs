/// Home of functions to compute the FFT.
///
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
/// + `data` - the sequence of N complex values with real parts
///    even-numbered entries and imaginary parts in odd entries.
///
/// + `inverse` - if true will compute the inverse transform;
///    the convention here is to put the 1/N normalizing factor
///    on the inverse transform.
///
pub fn fft(data: &mut [f64], inverse: bool) {
  let data_len = data.len() / 2;
  if (data_len as f64).log2().fract() != 0.0 {
    panic!("FFT routine is only implemented for arrays with length a power of 2.")
  }

  let n = data.len() / 2;
  let nn = n << 1;

  // Phase sign and normalizing factor for inverse transform.
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

    // Compute next potential swap row.
    m = n;
    while m >= 2 && j > m {
      j -= m;
      m = m >> 1;
    }
    j += m;

    // Divide by complex sequence length if inverse.
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

    // Start recurrence for trigonometric functions
    // of whole multiples of a given angle.
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
