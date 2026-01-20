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
}
