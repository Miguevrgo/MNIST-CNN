use crate::{layer::Layer, tensor::Tensor};

pub struct Network {
    pub layers: Vec<Box<dyn Layer>>,
    training: bool,
}

impl Network {
    pub fn new() -> Self {
        Network {
            layers: Vec::new(),
            training: false,
        }
    }

    fn add<L: Layer + 'static>(&mut self, layer: L) {
        self.layers.push(Box::new(layer))
    }

    fn forward(&mut self, output: &Tensor) -> Tensor {
        let mut grad = output.clone();
        for layer in &mut self.layers {
            grad = layer.forward(&grad)
        }
        grad
    }
}
