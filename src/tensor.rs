use std::ops::{Index, IndexMut};

use rand_distr::{Distribution, Normal};

/// A multi-dimensional tensor for neural network computations
///
/// //TODO: ?
/// Stored in row-major order with shape [height, witdth, depth, num_filters]
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

    pub fn he(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let fan_in: usize = shape.iter().skip(1).product();
        let normal_distr = Normal::new(0.0, (2.0 / fan_in as f32).sqrt()).unwrap();

        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| normal_distr.sample(&mut rng)).collect();

        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn xavier(shape: &[usize]) -> Self {
        let size = shape.iter().product();
        let fan_in = if shape.len() >= 2 {
            shape[shape.len() - 1]
        } else {
            1
        };
        let fan_out = if shape.len() >= 2 {
            shape[shape.len() - 2]
        } else {
            1
        };
        let std_deviation = (2.0 / (fan_in + fan_out) as f32).sqrt();
        let normal_distr = Normal::new(0.0, std_deviation).unwrap();
        let mut rng = rand::rng();
        let data: Vec<f32> = (0..size).map(|_| normal_distr.sample(&mut rng)).collect();

        Tensor {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Result<Tensor, &'static str> {
        let n_size = new_shape.iter().product();
        if self.data.len() != n_size {
            return Err("New shape must have the same number of total elements");
        }

        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape.to_vec(),
        })
    }

    pub fn flatten(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
            shape: vec![self.data.len()],
        }
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.shape != other.shape {
            return Err("Shapes must match for addition");
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.shape != other.shape {
            return Err("Shapes must match for substraction");
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    /// Performs the Hadamard product
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.shape != other.shape {
            return Err("Shapes must match for multiplication");
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    pub fn div(&self, other: &Tensor) -> Result<Tensor, &'static str> {
        if self.shape != other.shape {
            return Err("Shapes must match for substraction");
        }
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect();
        Ok(Tensor {
            data,
            shape: self.shape.clone(),
        })
    }

    pub fn scale(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|&elem| elem * scalar).collect();

        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn sum(&self) -> f32 {
        return self.data.iter().sum();
    }

    pub fn map<F>(&self, f: F) -> Tensor
    where
        F: Fn(f32) -> f32,
    {
        let data: Vec<f32> = self.data.iter().map(|&x| f(x)).collect();
        Tensor {
            data,
            shape: self.shape.clone(),
        }
    }

    pub fn relu(&self) -> Tensor {
        self.map(|x| x.max(0.0))
    }

    /// Clipped ReLU
    pub fn crelu(&self) -> Tensor {
        self.map(|x| x.max(0.0).min(1.0))
    }

    /// Squared Crelu
    pub fn screlu(&self) -> Tensor {
        self.map(|x| x.max(0.0).min(1.0).powi(2))
    }

    pub fn tanh(&self) -> Tensor {
        self.map(|x| x.tanh())
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
