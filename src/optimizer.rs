use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self, params: &mut [&mut Tensor], grads: &[&Tensor]);
}
