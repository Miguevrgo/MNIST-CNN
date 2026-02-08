use crate::tensor::Tensor;

pub trait Layer {
    // Forward pass
    fn forward(&mut self, input: &Tensor) -> Tensor;

    fn backward(&mut self, grad_output: &Tensor) -> Tensor;
}

/// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
/// Conv2D -- 2D Convolution with optional FFT-based computation
/// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
///
/// Input:  [batch, in_channels, H, W]
/// Output: [batch, out_channels, H_out, W_out]
/// Weights: [out_channels, in_channels, kernel_size, kernel_size]
///
/// Output spatial size: (W - F + 2P) / S + 1
///
/// The FFT const generic selects between:
///   FFT=true  -> frequency-domain convolution (pad to power-of-2, FFT, multiply, IFFT)
///   FFT=false -> direct spatial convolution (im2col-style loops)
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
    input_cache: Option<Tensor>,
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
            input_cache: None,
        }
    }

    fn output_size(&self, input_size: usize) -> usize {
        (input_size + 2 * self.padding - self.kernel_size) / self.stride + 1
    }

    /// Get pixel from input with padding (returns 0 for out-of-bounds)
    #[inline]
    fn get_padded(input: &[f32], h: isize, w: isize, ih: usize, iw: usize) -> f32 {
        if h < 0 || w < 0 || h >= ih as isize || w >= iw as isize {
            0.0
        } else {
            input[h as usize * iw + w as usize]
        }
    }
}

impl Layer for Conv2D<false> {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_cache = Some(input.clone());
        // TODO:
        unimplemented!()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        //TODO:
        unimplemented!()
    }
}

pub struct ReLU {
    input_cache: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { input_cache: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_cache = Some(input.clone());
        input.relu()
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self.input_cache.as_ref().unwrap();
        let mask = input.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
        grad_output.mul(&mask).unwrap()
    }
}
