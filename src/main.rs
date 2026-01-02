use std::f32::consts::PI;

// Search for complex number crates in rust or implement one
// FFT algorithm implementation
// Loading EMNIST
// CNN
// CNN improved using AVX extensions &&/|| GPU
use num_complex::Complex;

/// Performs the FFT and IFFT over some polynomial using the
/// Cooley-Tukey algorithm radix-2
fn fft<const INVERSE: bool>(
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

fn main() -> Result<(), &'static str> {
    let pol_1 = [3, 4, 5, 5, 2, 3, 0, 7];
    let pol_2 = [5, 2, 3, 1, 6, 0, 2, 7];

    let target_size = (pol_1.len() + pol_2.len() - 1).next_power_of_two();

    let mut c1: Vec<_> = pol_1.iter().map(|&v| Complex::new(v as f32, 0.0)).collect();
    let mut c2: Vec<_> = pol_2.iter().map(|&v| Complex::new(v as f32, 0.0)).collect();

    c1.resize(target_size, Complex::new(0.0, 0.0));
    c2.resize(target_size, Complex::new(0.0, 0.0));

    let f1 = fft::<false>(c1)?;
    let f2 = fft::<false>(c2)?;

    let multiplied: Vec<_> = f1.iter().zip(f2.iter()).map(|(a, b)| a * b).collect();

    let inv = fft::<true>(multiplied)?;

    let n = target_size as f32;
    let result: Vec<_> = inv
        .iter()
        .take(pol_1.len() + pol_2.len() - 1)
        .map(|c| (c.re / n).round() as i32)
        .collect();

    println!("{:?}", result);
    Ok(())
}
