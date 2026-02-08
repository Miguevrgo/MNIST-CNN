use crate::tensor::Tensor;

/// Optimizer trait: updates parameters given their gradients.
pub trait Optimizer {
    fn step(&mut self, params: &mut [(&mut Tensor, &Tensor)]);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// SGD with optional momentum
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// v_t = momentum * v_{t-1} + grad
// param = param - lr * v_t

pub struct SGD {
    pub lr: f32,
    pub momentum: f32,
    velocities: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32) -> Self {
        Self {
            lr,
            momentum,
            velocities: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, params: &mut [(&mut Tensor, &Tensor)]) {
        // Initialize velocities on first call
        if self.velocities.len() != params.len() {
            self.velocities = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
        }

        for (i, (param, grad)) in params.iter_mut().enumerate() {
            for j in 0..param.len() {
                self.velocities[i][j] = self.momentum * self.velocities[i][j] + grad.data[j];
                param.data[j] -= self.lr * self.velocities[i][j];
            }
        }
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AdamW (Adam with decoupled weight decay)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//
// m_t = beta1 * m_{t-1} + (1 - beta1) * grad
// v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
// m_hat = m_t / (1 - beta1^t)
// v_hat = v_t / (1 - beta2^t)
// param = param * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)

pub struct AdamW {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    t: usize,
    m: Vec<Vec<f32>>, // First moment estimates
    v: Vec<Vec<f32>>, // Second moment estimates
}

impl AdamW {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }

    pub fn default_with_lr(lr: f32) -> Self {
        Self::new(lr, 0.9, 0.999, 1e-8, 0.01)
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, params: &mut [(&mut Tensor, &Tensor)]) {
        // Initialize moments on first call
        if self.m.len() != params.len() {
            self.m = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
            self.v = params.iter().map(|(p, _)| vec![0.0; p.len()]).collect();
        }

        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32); // bias correction
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, (param, grad)) in params.iter_mut().enumerate() {
            for j in 0..param.len() {
                // Update moments
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad.data[j];
                self.v[i][j] =
                    self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad.data[j] * grad.data[j];

                // Bias-corrected estimates
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;

                // Decoupled weight decay + Adam update
                param.data[j] *= 1.0 - self.lr * self.weight_decay;
                param.data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}
