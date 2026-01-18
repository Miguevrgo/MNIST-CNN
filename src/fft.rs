// Search for complex number crates in rust or implement one
// FFT algorithm implementation
// Loading EMNIST
// CNN
// CNN improved using AVX extensions &&/|| GPU
use num_complex::Complex;
use std::f32::consts::PI;

/// Performs the FFT and IFFT over some polynomial using the
/// Cooley-Tukey algorithm radix-2
pub fn fft<const INVERSE: bool>(
    polynomial: Vec<Complex<f32>>,
) -> Result<Vec<Complex<f32>>, &'static str> {
    let n = polynomial.len();

    if n.count_ones() != 1 {
        return Err("FFT only supported for powers of 2");
    }

    if n == 1 {
        return Ok(polynomial);
    }

    let mut poly_even: Vec<_> = polynomial.iter().step_by(2).copied().collect();
    let mut poly_odd: Vec<_> = polynomial.iter().skip(1).step_by(2).copied().collect();

    let sign = if INVERSE { -1.0 } else { 1.0 };
    let w_n = Complex::cis(sign * 2.0 * PI / n as f32);

    poly_even = fft::<INVERSE>(poly_even)?;
    poly_odd = fft::<INVERSE>(poly_odd)?;

    let mut result = vec![Complex::from(0.0); n];
    let mut w = Complex::from(1.0);
    for j in 0..n / 2 {
        let t = w * poly_odd[j];
        result[j] = poly_even[j] + t;
        result[j + n / 2] = poly_even[j] - t;
        w *= w_n;
    }

    Ok(result)
}
