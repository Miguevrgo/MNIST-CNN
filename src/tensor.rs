use std::ops::{Index, IndexMut};

use rand_distr::{Distribution, Normal};

/// A multi-dimensional tensor for neural network computations.
/// Stored in row-major (C-contiguous) order.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![0.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn ones(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        Self {
            data: vec![1.0; size],
            shape: shape.to_vec(),
        }
    }

    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>());
        Self {
            data: data.to_vec(),
            shape: shape.to_vec(),
        }
    }

    /// He (Kaiming) initialization: N(0, sqrt(2/fan_in))
    pub fn he(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let fan_in: usize = shape.iter().skip(1).product();
        let normal = Normal::new(0.0, (2.0 / fan_in as f32).sqrt()).unwrap();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    /// Xavier (Glorot) initialization: N(0, sqrt(2/(fan_in+fan_out)))
    pub fn xavier(shape: &[usize]) -> Self {
        let size: usize = shape.iter().product();
        let fan_in = if shape.len() >= 2 { shape[1] } else { 1 };
        let fan_out = if shape.len() >= 2 { shape[0] } else { 1 };
        let std_dev = (2.0 / (fan_in + fan_out) as f32).sqrt();
        let normal = Normal::new(0.0, std_dev).unwrap();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng)).collect();
        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
        debug_assert_eq!(
            self.data.len(),
            new_shape.iter().product::<usize>(),
            "Reshape: element count mismatch"
        );
        Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        }
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: vec![self.data.len()],
        }
    }

    // ── Element-wise operations ──

    pub fn add(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.shape, other.shape, "add: shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.shape, other.shape, "sub: shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Hadamard (element-wise) product
    pub fn mul(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.shape, other.shape, "mul: shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.shape, other.shape, "div: shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a / b)
            .collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x * scalar).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    /// Add scalar to all elements
    pub fn add_scalar(&self, s: f32) -> Tensor {
        self.map(|x| x + s)
    }

    /// Square root of each element
    pub fn sqrt(&self) -> Tensor {
        self.map(|x| x.sqrt())
    }

    // ── Activation functions ──

    pub fn relu(&self) -> Tensor {
        self.map(|x| x.max(0.0))
    }

    /// Clipped ReLU: clamp to [0, 1]
    pub fn crelu(&self) -> Tensor {
        self.map(|x| x.max(0.0).min(1.0))
    }

    /// Squared Clipped ReLU
    pub fn screlu(&self) -> Tensor {
        self.map(|x| x.max(0.0).min(1.0).powi(2))
    }

    pub fn tanh_act(&self) -> Tensor {
        self.map(|x| x.tanh())
    }

    // ── Matrix multiplication ──

    /// Matrix multiply: self [M, K] x other [K, N] -> [M, N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(self.shape.len(), 2, "matmul: self must be 2D");
        debug_assert_eq!(other.shape.len(), 2, "matmul: other must be 2D");
        let m = self.shape[0];
        let k = self.shape[1];
        debug_assert_eq!(k, other.shape[0], "matmul: inner dimensions mismatch");
        let n = other.shape[1];

        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for p in 0..k {
                let a = self.data[i * k + p];
                for j in 0..n {
                    data[i * n + j] += a * other.data[p * n + j];
                }
            }
        }
        Tensor {
            data,
            shape: vec![m, n],
        }
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Tensor {
        debug_assert_eq!(self.shape.len(), 2, "transpose: must be 2D");
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }
        Tensor {
            data,
            shape: vec![cols, rows],
        }
    }

    // ── Softmax & argmax ──

    /// Softmax over a 2D tensor [batch, classes], computed per row.
    /// Numerically stable: subtract max per row before exp.
    pub fn softmax(&self) -> Tensor {
        debug_assert_eq!(self.shape.len(), 2, "softmax: must be 2D [batch, classes]");
        let (batch, classes) = (self.shape[0], self.shape[1]);
        let mut data = vec![0.0; batch * classes];
        for b in 0..batch {
            let row = &self.data[b * classes..(b + 1) * classes];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for c in 0..classes {
                data[b * classes + c] = exps[c] / sum;
            }
        }
        Tensor {
            data,
            shape: vec![batch, classes],
        }
    }

    /// Argmax over a 1D or 2D tensor. If 2D [batch, classes], returns
    /// index of max per row.
    pub fn argmax(&self) -> Vec<usize> {
        if self.shape.len() == 1 {
            let (idx, _) = self
                .data
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            return vec![idx];
        }
        let (batch, cols) = (self.shape[0], self.shape[1]);
        (0..batch)
            .map(|b| {
                let row = &self.data[b * cols..(b + 1) * cols];
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect()
    }
}

impl Index<usize> for Tensor {
    type Output = f32;
    fn index(&self, idx: usize) -> &Self::Output {
        &self.data[idx]
    }
}

impl IndexMut<usize> for Tensor {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.data[idx]
    }
}
