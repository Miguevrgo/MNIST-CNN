use crate::{
    layer::Conv2D,
    mnist::{MnistDataset, download_instructions},
    network::Network,
};
use num_complex::Complex;
use std::path::Path;

mod fft;
mod layer;
mod mnist;
mod network;
mod optimizer;
mod tensor;

const WELCOME_STRING: &str = "
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MNIST-CNN with FFT-based Convolutions                     ║
║                              Author: Miguevrgo                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
";
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 0.01;
const MOMENTUM: f32 = 0.9;

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

    // ── Build network ──
    // Architecture: Conv(1->8, 3x3) -> ReLU -> MaxPool(2)
    //             -> Conv(8->16, 3x3) -> ReLU -> MaxPool(2)
    //             -> Flatten -> Dense(400, 128) -> ReLU -> Dense(128, 10)
    //
    // Input:  [B, 1, 28, 28]
    // Conv1:  [B, 8, 26, 26]  (28-3+1=26)
    // Pool1:  [B, 8, 13, 13]
    // Conv2:  [B, 16, 11, 11] (13-3+1=11)
    // Pool2:  [B, 16, 5, 5]   (11/2=5)
    // Flat:   [B, 400]
    // Dense1: [B, 128]
    // Dense2: [B, 10]
    //
    // Conv2D<true> uses FFT convolution, Conv2D<false> uses direct convolution.
    // For small 3x3 kernels on 28x28 images, direct is faster, but FFT is here
    // to demonstrate the algorithm. Use Conv2D<true> for the first layer as a demo.

    println!("[+] Building network...");
    let mut network = Network::new();

    network.add(Conv2D::<true>::new(1, 8, 3, 1, 0));
    network.add(ReLu::new());
}
