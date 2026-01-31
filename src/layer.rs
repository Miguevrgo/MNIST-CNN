use crate::tensor::Tensor;

pub trait Layer {
    // Forward pass
    fn forward(&mut self, input: &Tensor) -> Tensor;

    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
}

/// Convolutional Layer for 2D kernel using optional FFT algorithm for
/// convolutions calculations, alternative is im2col. This logic was
/// generated with some help from LLM + this resource:
/// https://cs231n.github.io/convolutional-networks/ which for me was
/// perfectly explained and a great way to understand the net.
///
/// (W - F + 2P) / S + 1
/// W -> Input Volume size
/// F -> Receptive field size
/// S -> Stride
/// P -> Zero-padding
///
pub struct Conv2D<const FFT: bool> {
    pub depth: usize,       // in_channels
    pub num_filters: usize, // out_channels
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,

    pub weights: Tensor, // [kernel_size, kernel_size, depth, num_filters]
    pub bias: Tensor,    // [num_filters]

    pub grad_weights: Tensor,
    pub grad_bias: Tensor,
}

impl Conv2D<true> {
    pub fn new<const FFT: bool>(
        depth: usize,
        num_filters: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let weights = Tensor::he(&[kernel_size, kernel_size, depth, num_filters]);
        let bias = Tensor::zeros(&[num_filters]);
        let grad_weights = Tensor::zeros(&[kernel_size, kernel_size, depth, num_filters]);
        let grad_bias = Tensor::zeros(&[num_filters]);
        Self {
            depth,
            num_filters,
            kernel_size,
            stride,
            padding,
            weights,
            bias,
            grad_weights,
            grad_bias,
        }
    }
}
