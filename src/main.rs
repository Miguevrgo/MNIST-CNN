use crate::{
    mnist::{MnistDataset, download_instructions},
    network::Network,
};
use num_complex::Complex;
use std::path::Path;

mod fft;
mod mnist;
mod network;

const WELCOME_STRING: &str = "
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MNIST-CNN with FFT-based Convolutions                     ║
║                              Author: Miguevrgo                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
";

fn main() {
    println!("\x1b[1;33m {WELCOME_STRING} \x1b[0m");
    println!("[+] Loading MNIST dataset...");

    let training_data = match MnistDataset::load::<true>(Path::new("./data")) {
        Ok(data) => {
            println!("\t Training samples: {}", data.num_samples);
            data
        }
        Err(e) => {
            eprintln!("\t Error loading training data: {e}");
            eprintln!("{}", download_instructions());
            return;
        }
    };

    let test_data = match MnistDataset::load::<false>(Path::new("./data")) {
        Ok(data) => {
            println!("\t Test samples: {}", data.num_samples);
            data
        }
        Err(e) => {
            eprintln!("\t Error loading test data: {e}");
            eprintln!("{}", download_instructions());
            return;
        }
    };

    println!("[+] Creating neural network");
    let mut network = Network::new();

    let pol_1 = [3, 4, 5, 5, 2, 3, 0, 7];
    let pol_2 = [5, 2, 3, 1, 6, 0, 2, 7];

    let target_size = (pol_1.len() + pol_2.len() - 1).next_power_of_two();

    let mut c1: Vec<_> = pol_1.iter().map(|&v| Complex::new(v as f32, 0.0)).collect();
    let mut c2: Vec<_> = pol_2.iter().map(|&v| Complex::new(v as f32, 0.0)).collect();

    c1.resize(target_size, Complex::new(0.0, 0.0));
    c2.resize(target_size, Complex::new(0.0, 0.0));

    let f1 = fft::fft::<false>(c1).expect("Unable to compute fft for pol_1");
    let f2 = fft::fft::<false>(c2).expect("Unable to compute fft for pol_2");

    let multiplied: Vec<_> = f1.iter().zip(f2.iter()).map(|(a, b)| a * b).collect();

    let inv = fft::fft::<true>(multiplied).expect("Unable to compute inverse fft");

    let n = target_size as f32;
    let result: Vec<_> = inv
        .iter()
        .take(pol_1.len() + pol_2.len() - 1)
        .map(|c| (c.re / n).round() as i32)
        .collect();

    println!("{:?}", result);
}
