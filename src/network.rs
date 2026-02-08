use crate::{layer::Layer, tensor::Tensor};

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Self {
        Network { layers: Vec::new() }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut out = input.clone();
        for layer in &mut self.layers {
            out = layer.forward(&out);
        }
        out
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }

    /// Collects all (param, grad) pairs from every layer for the optimizer
    pub fn params_and_grads(&mut self) -> Vec<(&mut Tensor, &Tensor)> {
        let mut all = Vec::new();
        for layer in &mut self.layers {
            all.extend(layer.params_and_grads());
        }
        all
    }
}
