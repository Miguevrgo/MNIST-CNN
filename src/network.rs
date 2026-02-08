use crate::{layer::Layer, tensor::Tensor};

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn new() -> Self {
        Network { layers: Vec::new() }
    }

    pub fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer))
    }

    pub fn forward(&mut self, output: &Tensor) -> Tensor {
        let mut grad = output.clone();
        for layer in &mut self.layers {
            grad = layer.forward(&grad)
        }
        grad
    }

    pub fn backward(&mut self, grad_output: &Tensor) -> Tensor {
        let mut grad = grad_output.clone();
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
        grad
    }
}
