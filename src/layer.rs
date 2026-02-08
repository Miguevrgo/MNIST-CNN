use crate::fft;
use crate::tensor::Tensor;
use num_complex::Complex;

/// Every layer must implement forward, backward, and provide
/// mutable access to its parameters and gradients for the optimizer.
pub trait Layer {
    fn forward(&mut self, input: &Tensor) -> Tensor;
    fn backward(&mut self, grad_output: &Tensor) -> Tensor;

    /// Returns (params, grads) pairs for optimizer updates.
    /// Layers without learnable parameters return empty vecs.
    fn params_and_grads(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        vec![]
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Conv2D -- 2D Convolution with optional FFT-based computation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Input:  [batch, in_channels, H, W]
// Output: [batch, out_channels, H_out, W_out]
// Weights: [out_channels, in_channels, kernel_size, kernel_size]
//
// Output spatial size: (W - F + 2P) / S + 1
//
// The FFT const generic selects between:
//   FFT=true  -> frequency-domain convolution (pad to power-of-2, FFT, multiply, IFFT)
//   FFT=false -> direct spatial convolution (im2col-style loops)

pub struct Conv2D<const FFT: bool> {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,

    pub weights: Tensor, // [out_channels, in_channels, K, K]
    pub bias: Tensor,    // [out_channels]
    pub grad_weights: Tensor,
    pub grad_bias: Tensor,

    // Cached for backward pass
    input_cache: Option<Tensor>,
}

impl<const FFT: bool> Conv2D<FFT> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let weights = Tensor::he(&[out_channels, in_channels, kernel_size, kernel_size]);
        let bias = Tensor::zeros(&[out_channels]);
        let grad_weights = Tensor::zeros(&[out_channels, in_channels, kernel_size, kernel_size]);
        let grad_bias = Tensor::zeros(&[out_channels]);
        Self {
            in_channels,
            out_channels,
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

// ── Direct (spatial) convolution ──

impl Layer for Conv2D<false> {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_cache = Some(input.clone());
        let (batch, _ic, ih, iw) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let oh = self.output_size(ih);
        let ow = self.output_size(iw);
        let k = self.kernel_size;
        let mut output = Tensor::zeros(&[batch, self.out_channels, oh, ow]);

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut val = self.bias.data[oc];
                        for ic in 0..self.in_channels {
                            let in_offset = (b * self.in_channels + ic) * ih * iw;
                            let w_offset = (oc * self.in_channels + ic) * k * k;
                            for ki in 0..k {
                                for kj in 0..k {
                                    let h = (i * self.stride + ki) as isize - self.padding as isize;
                                    let w = (j * self.stride + kj) as isize - self.padding as isize;
                                    val += Self::get_padded(&input.data[in_offset..], h, w, ih, iw)
                                        * self.weights.data[w_offset + ki * k + kj];
                                }
                            }
                        }
                        output.data[(b * self.out_channels + oc) * oh * ow + i * ow + j] = val;
                    }
                }
            }
        }
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self
            .input_cache
            .as_ref()
            .expect("Must call forward before backward");
        let (batch, _ic, ih, iw) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let oh = grad_output.shape[2];
        let ow = grad_output.shape[3];
        let k = self.kernel_size;

        // Zero gradients
        self.grad_weights.data.fill(0.0);
        self.grad_bias.data.fill(0.0);
        let mut grad_input = Tensor::zeros(&input.shape);

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for i in 0..oh {
                    for j in 0..ow {
                        let g =
                            grad_output.data[(b * self.out_channels + oc) * oh * ow + i * ow + j];
                        self.grad_bias.data[oc] += g;

                        for ic in 0..self.in_channels {
                            let in_offset = (b * self.in_channels + ic) * ih * iw;
                            let w_offset = (oc * self.in_channels + ic) * k * k;
                            for ki in 0..k {
                                for kj in 0..k {
                                    let h = (i * self.stride + ki) as isize - self.padding as isize;
                                    let w = (j * self.stride + kj) as isize - self.padding as isize;
                                    if h >= 0 && w >= 0 && (h as usize) < ih && (w as usize) < iw {
                                        let idx = in_offset + h as usize * iw + w as usize;
                                        self.grad_weights.data[w_offset + ki * k + kj] +=
                                            input.data[idx] * g;
                                        grad_input.data[idx] +=
                                            self.weights.data[w_offset + ki * k + kj] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        grad_input
    }

    fn params_and_grads(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        // SAFETY: we need two simultaneous references into self. We use raw pointers
        // because the fields are disjoint. This is a common pattern in Rust ML code.
        let gw = &self.grad_weights as *const Tensor;
        let gb = &self.grad_bias as *const Tensor;
        unsafe { vec![(&mut self.weights, &*gw), (&mut self.bias, &*gb)] }
    }
}

// ── FFT-based convolution ──

impl Layer for Conv2D<true> {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_cache = Some(input.clone());
        let (batch, _ic, ih, iw) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let oh = self.output_size(ih);
        let ow = self.output_size(iw);
        let k = self.kernel_size;

        // For FFT we need stride=1 -- fall back to direct if stride != 1
        // (FFT convolution naturally computes stride=1)
        if self.stride != 1 {
            // Delegate to direct convolution for non-unit strides
            let mut direct = Conv2D::<false> {
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                weights: self.weights.clone(),
                bias: self.bias.clone(),
                grad_weights: self.grad_weights.clone(),
                grad_bias: self.grad_bias.clone(),
                input_cache: self.input_cache.clone(),
            };
            return direct.forward(input);
        }

        // Padded input size and FFT size (must be power of 2)
        let padded_h = ih + 2 * self.padding;
        let padded_w = iw + 2 * self.padding;
        let fft_h = (padded_h + k - 1).next_power_of_two();
        let fft_w = (padded_w + k - 1).next_power_of_two();
        let fft_size = fft_h * fft_w;

        let mut output = Tensor::zeros(&[batch, self.out_channels, oh, ow]);

        for b in 0..batch {
            for oc in 0..self.out_channels {
                // Accumulate over input channels in frequency domain
                let mut acc = vec![Complex::new(0.0f32, 0.0); fft_size];

                for ic in 0..self.in_channels {
                    // Pad input channel into fft_h x fft_w
                    let mut input_buf = vec![Complex::new(0.0f32, 0.0); fft_size];
                    let in_offset = (b * self.in_channels + ic) * ih * iw;
                    for r in 0..padded_h {
                        for c in 0..padded_w {
                            let src_r = r as isize - self.padding as isize;
                            let src_c = c as isize - self.padding as isize;
                            if src_r >= 0
                                && src_c >= 0
                                && (src_r as usize) < ih
                                && (src_c as usize) < iw
                            {
                                input_buf[r * fft_w + c] = Complex::new(
                                    input.data[in_offset + src_r as usize * iw + src_c as usize],
                                    0.0,
                                );
                            }
                        }
                    }

                    // Pad kernel into fft_h x fft_w
                    let mut kernel_buf = vec![Complex::new(0.0f32, 0.0); fft_size];
                    let w_offset = (oc * self.in_channels + ic) * k * k;
                    for r in 0..k {
                        for c in 0..k {
                            kernel_buf[r * fft_w + c] =
                                Complex::new(self.weights.data[w_offset + r * k + c], 0.0);
                        }
                    }

                    // 2D FFT via row-then-column 1D FFTs
                    let input_freq = fft_2d::<false>(&input_buf, fft_h, fft_w);
                    let kernel_freq = fft_2d::<false>(&kernel_buf, fft_h, fft_w);

                    // Pointwise multiply and accumulate
                    for i in 0..fft_size {
                        acc[i] += input_freq[i] * kernel_freq[i];
                    }
                }

                // IFFT to get spatial result
                let result = fft_2d::<true>(&acc, fft_h, fft_w);
                let norm = (fft_h * fft_w) as f32;

                // Extract valid region and add bias
                for i in 0..oh {
                    for j in 0..ow {
                        let val = result[i * fft_w + j].re / norm + self.bias.data[oc];
                        output.data[(b * self.out_channels + oc) * oh * ow + i * ow + j] = val;
                    }
                }
            }
        }
        output
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        // Backward pass uses direct computation (FFT backward is complex and
        // the performance benefit is minimal for small MNIST kernels)
        let input = self
            .input_cache
            .as_ref()
            .expect("Must call forward before backward");
        let (batch, _ic, ih, iw) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        let oh = grad_output.shape[2];
        let ow = grad_output.shape[3];
        let k = self.kernel_size;

        self.grad_weights.data.fill(0.0);
        self.grad_bias.data.fill(0.0);
        let mut grad_input = Tensor::zeros(&input.shape);

        for b in 0..batch {
            for oc in 0..self.out_channels {
                for i in 0..oh {
                    for j in 0..ow {
                        let g =
                            grad_output.data[(b * self.out_channels + oc) * oh * ow + i * ow + j];
                        self.grad_bias.data[oc] += g;

                        for ic in 0..self.in_channels {
                            let in_off = (b * self.in_channels + ic) * ih * iw;
                            let w_off = (oc * self.in_channels + ic) * k * k;
                            for ki in 0..k {
                                for kj in 0..k {
                                    let h = (i * self.stride + ki) as isize - self.padding as isize;
                                    let w = (j * self.stride + kj) as isize - self.padding as isize;
                                    if h >= 0 && w >= 0 && (h as usize) < ih && (w as usize) < iw {
                                        let idx = in_off + h as usize * iw + w as usize;
                                        self.grad_weights.data[w_off + ki * k + kj] +=
                                            input.data[idx] * g;
                                        grad_input.data[idx] +=
                                            self.weights.data[w_off + ki * k + kj] * g;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        grad_input
    }

    fn params_and_grads(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let gw = &self.grad_weights as *const Tensor;
        let gb = &self.grad_bias as *const Tensor;
        unsafe { vec![(&mut self.weights, &*gw), (&mut self.bias, &*gb)] }
    }
}

/// 2D FFT computed as row-wise 1D FFTs followed by column-wise 1D FFTs
fn fft_2d<const INVERSE: bool>(
    data: &[Complex<f32>],
    rows: usize,
    cols: usize,
) -> Vec<Complex<f32>> {
    let mut buf = data.to_vec();

    // Row-wise FFT
    for r in 0..rows {
        let row: Vec<_> = buf[r * cols..(r + 1) * cols].to_vec();
        let transformed = fft::fft::<INVERSE>(row).expect("FFT failed");
        buf[r * cols..(r + 1) * cols].copy_from_slice(&transformed);
    }

    // Column-wise FFT
    for c in 0..cols {
        let col: Vec<_> = (0..rows).map(|r| buf[r * cols + c]).collect();
        let transformed = fft::fft::<INVERSE>(col).expect("FFT failed");
        for r in 0..rows {
            buf[r * cols + c] = transformed[r];
        }
    }

    buf
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Dense (Fully Connected) Layer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// Input:  [batch, in_features]
// Output: [batch, out_features]
// y = x * W^T + b

pub struct Dense {
    pub in_features: usize,
    pub out_features: usize,
    pub weights: Tensor, // [out_features, in_features]
    pub bias: Tensor,    // [out_features]
    pub grad_weights: Tensor,
    pub grad_bias: Tensor,
    input_cache: Option<Tensor>,
}

impl Dense {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            weights: Tensor::he(&[out_features, in_features]),
            bias: Tensor::zeros(&[out_features]),
            grad_weights: Tensor::zeros(&[out_features, in_features]),
            grad_bias: Tensor::zeros(&[out_features]),
            input_cache: None,
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        // input: [batch, in_features]
        let input_2d = if input.shape.len() == 1 {
            input.reshape(&[1, input.len()])
        } else {
            input.clone()
        };
        self.input_cache = Some(input_2d.clone());

        let batch = input_2d.shape[0];
        // W^T: [in_features, out_features]
        let wt = self.weights.transpose();
        let mut out = input_2d.matmul(&wt); // [batch, out_features]

        // Add bias to each row
        for b in 0..batch {
            for j in 0..self.out_features {
                out.data[b * self.out_features + j] += self.bias.data[j];
            }
        }
        out
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input = self
            .input_cache
            .as_ref()
            .expect("Must call forward before backward");
        let batch = input.shape[0];

        // grad_weights = grad_output^T * input  -> [out, in]
        let got = grad_output.reshape(&[batch, self.out_features]);
        self.grad_weights = got.transpose().matmul(input);

        // grad_bias = sum over batch
        self.grad_bias.data.fill(0.0);
        for b in 0..batch {
            for j in 0..self.out_features {
                self.grad_bias.data[j] += got.data[b * self.out_features + j];
            }
        }

        // grad_input = grad_output * W  -> [batch, in_features]
        got.matmul(&self.weights)
    }

    fn params_and_grads(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let gw = &self.grad_weights as *const Tensor;
        let gb = &self.grad_bias as *const Tensor;
        unsafe { vec![(&mut self.weights, &*gw), (&mut self.bias, &*gb)] }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ReLU Activation Layer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

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
        grad_output.mul(&mask)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Flatten Layer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Converts [batch, C, H, W] -> [batch, C*H*W]

pub struct Flatten {
    input_shape: Option<Vec<usize>>,
}

impl Flatten {
    pub fn new() -> Self {
        Self { input_shape: None }
    }
}

impl Layer for Flatten {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        self.input_shape = Some(input.shape.clone());
        let batch = input.shape[0];
        let flat_size: usize = input.shape[1..].iter().product();
        input.reshape(&[batch, flat_size])
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let shape = self.input_shape.as_ref().unwrap();
        grad_output.reshape(shape)
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MaxPool2D Layer
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Input:  [batch, channels, H, W]
// Output: [batch, channels, H/pool, W/pool]

pub struct MaxPool2D {
    pub pool_size: usize,
    max_indices: Option<Vec<usize>>, // index of max element for backward
    input_shape: Option<Vec<usize>>,
}

impl MaxPool2D {
    pub fn new(pool_size: usize) -> Self {
        Self {
            pool_size,
            max_indices: None,
            input_shape: None,
        }
    }
}

impl Layer for MaxPool2D {
    fn forward(&mut self, input: &Tensor) -> Tensor {
        let (batch, channels, ih, iw) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );
        self.input_shape = Some(input.shape.clone());
        let oh = ih / self.pool_size;
        let ow = iw / self.pool_size;
        let out_size = batch * channels * oh * ow;
        let mut output = vec![0.0f32; out_size];
        let mut indices = vec![0usize; out_size];

        for b in 0..batch {
            for c in 0..channels {
                let in_base = (b * channels + c) * ih * iw;
                let out_base = (b * channels + c) * oh * ow;
                for i in 0..oh {
                    for j in 0..ow {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_idx = 0;
                        for pi in 0..self.pool_size {
                            for pj in 0..self.pool_size {
                                let r = i * self.pool_size + pi;
                                let c_idx = j * self.pool_size + pj;
                                let idx = in_base + r * iw + c_idx;
                                if input.data[idx] > max_val {
                                    max_val = input.data[idx];
                                    max_idx = idx;
                                }
                            }
                        }
                        output[out_base + i * ow + j] = max_val;
                        indices[out_base + i * ow + j] = max_idx;
                    }
                }
            }
        }

        self.max_indices = Some(indices);
        Tensor {
            data: output,
            shape: vec![batch, channels, oh, ow],
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let input_shape = self.input_shape.as_ref().unwrap();
        let indices = self.max_indices.as_ref().unwrap();
        let mut grad_input = Tensor::zeros(input_shape);

        for (i, &idx) in indices.iter().enumerate() {
            grad_input.data[idx] += grad_output.data[i];
        }
        grad_input
    }
}
