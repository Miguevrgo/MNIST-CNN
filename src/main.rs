use num_complex::Complex;

mod fft;
mod mnist;

fn main() -> Result<(), &'static str> {
    let pol_1 = [3, 4, 5, 5, 2, 3, 0, 7];
    let pol_2 = [5, 2, 3, 1, 6, 0, 2, 7];

    let target_size = (pol_1.len() + pol_2.len() - 1).next_power_of_two();

    let mut c1: Vec<_> = pol_1.iter().map(|&v| Complex::new(v as f32, 0.0)).collect();
    let mut c2: Vec<_> = pol_2.iter().map(|&v| Complex::new(v as f32, 0.0)).collect();

    c1.resize(target_size, Complex::new(0.0, 0.0));
    c2.resize(target_size, Complex::new(0.0, 0.0));

    let f1 = fft::fft::<false>(c1)?;
    let f2 = fft::fft::<false>(c2)?;

    let multiplied: Vec<_> = f1.iter().zip(f2.iter()).map(|(a, b)| a * b).collect();

    let inv = fft::fft::<true>(multiplied)?;

    let n = target_size as f32;
    let result: Vec<_> = inv
        .iter()
        .take(pol_1.len() + pol_2.len() - 1)
        .map(|c| (c.re / n).round() as i32)
        .collect();

    println!("{:?}", result);
    Ok(())
}
