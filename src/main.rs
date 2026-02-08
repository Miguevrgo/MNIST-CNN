#![allow(dead_code)]

use crate::{
    layer::{Conv2D, Dense, Flatten, MaxPool2D, ReLU},
    loss::cross_entropy_loss,
    mnist::{download_instructions, MnistDataset},
    network::Network,
    optimizer::{Optimizer, SGD},
    tensor::Tensor,
};
use std::path::Path;

mod fft;
mod layer;
mod loss;
mod mnist;
mod network;
mod optimizer;
mod tensor;

const WELCOME_STRING: &str = "
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MNIST-CNN with FFT-based Convolutions                    ║
║                              Author: Miguevrgo                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
";

// ── Hyperparameters ──
const BATCH_SIZE: usize = 32;
const EPOCHS: usize = 5;
const LEARNING_RATE: f32 = 0.01;
const MOMENTUM: f32 = 0.9;

fn main() {
    println!("\x1b[1;33m{WELCOME_STRING}\x1b[0m");

    // ── Load data ──
    println!("[+] Loading MNIST dataset...");
    let train = match MnistDataset::load::<true>(Path::new("./data")) {
        Ok(d) => {
            println!("\tTraining samples: {}", d.num_samples);
            d
        }
        Err(e) => {
            eprintln!("\tError: {e}\n{}", download_instructions());
            return;
        }
    };
    let test = match MnistDataset::load::<false>(Path::new("./data")) {
        Ok(d) => {
            println!("\tTest samples: {}", d.num_samples);
            d
        }
        Err(e) => {
            eprintln!("\tError: {e}\n{}", download_instructions());
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
    let mut net = Network::new();
    net.add(Conv2D::<true>::new(1, 8, 3, 1, 0)); // FFT convolution
    net.add(ReLU::new());
    net.add(MaxPool2D::new(2));
    net.add(Conv2D::<false>::new(8, 16, 3, 1, 0)); // Direct convolution
    net.add(ReLU::new());
    net.add(MaxPool2D::new(2));
    net.add(Flatten::new());
    net.add(Dense::new(16 * 5 * 5, 128));
    net.add(ReLU::new());
    net.add(Dense::new(128, 10));

    let mut optimizer = SGD::new(LEARNING_RATE, MOMENTUM);

    // ── Training loop ──
    println!("[+] Training for {EPOCHS} epochs, batch size {BATCH_SIZE}\n");
    let num_batches = train.num_samples / BATCH_SIZE;

    for epoch in 0..EPOCHS {
        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;
        let mut total = 0usize;

        for batch_idx in 0..num_batches {
            let start = batch_idx * BATCH_SIZE;

            // Extract batch of images: [BATCH_SIZE, 1, 28, 28]
            let img_start = start * 1 * 28 * 28;
            let img_end = img_start + BATCH_SIZE * 1 * 28 * 28;
            let batch_images = Tensor::from_slice(
                &train.images.data[img_start..img_end],
                &[BATCH_SIZE, 1, 28, 28],
            );

            // Extract batch of labels
            let batch_labels = &train.labels.data[start..start + BATCH_SIZE];

            // Forward pass
            let logits = net.forward(&batch_images);

            // Loss + gradient
            let (loss, grad) = cross_entropy_loss(&logits, batch_labels);
            epoch_loss += loss;

            // Accuracy
            let preds = logits.argmax();
            for i in 0..BATCH_SIZE {
                if preds[i] == batch_labels[i] as usize {
                    correct += 1;
                }
                total += 1;
            }

            // Backward pass
            net.backward(&grad);

            // Optimizer step
            let mut pg = net.params_and_grads();
            optimizer.step(&mut pg);

            // Progress report every 100 batches
            if (batch_idx + 1) % 100 == 0 {
                println!(
                    "  Epoch {}/{} | Batch {}/{} | Loss: {:.4} | Acc: {:.2}%",
                    epoch + 1,
                    EPOCHS,
                    batch_idx + 1,
                    num_batches,
                    epoch_loss / (batch_idx + 1) as f32,
                    100.0 * correct as f32 / total as f32,
                );
            }
        }

        let avg_loss = epoch_loss / num_batches as f32;
        let train_acc = 100.0 * correct as f32 / total as f32;
        println!(
            "\n  Epoch {}/{} complete | Avg Loss: {:.4} | Train Acc: {:.2}%",
            epoch + 1,
            EPOCHS,
            avg_loss,
            train_acc
        );

        // ── Evaluate on test set ──
        let test_acc = evaluate(&mut net, &test);
        println!("  Test Accuracy: {:.2}%\n", test_acc);
    }

    println!("[+] Training complete!");
}

/// Evaluate accuracy on a dataset (processes in batches to avoid OOM)
fn evaluate(net: &mut Network, dataset: &MnistDataset) -> f32 {
    let mut correct = 0usize;
    let mut total = 0usize;
    let eval_batch = 100;
    let num_batches = dataset.num_samples / eval_batch;

    for b in 0..num_batches {
        let start = b * eval_batch;
        let img_start = start * 28 * 28;
        let img_end = img_start + eval_batch * 28 * 28;
        let images = Tensor::from_slice(
            &dataset.images.data[img_start..img_end],
            &[eval_batch, 1, 28, 28],
        );
        let labels = &dataset.labels.data[start..start + eval_batch];

        let logits = net.forward(&images);
        let preds = logits.argmax();
        for i in 0..eval_batch {
            if preds[i] == labels[i] as usize {
                correct += 1;
            }
            total += 1;
        }
    }

    100.0 * correct as f32 / total as f32
}
