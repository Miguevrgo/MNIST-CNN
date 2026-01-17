use std::path::Path;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

pub struct MnistDataset {
    pub images: Tensor,
    pub labels: Tensor,
    pub num_samples: usize,
}

impl MnistDataset {
    pub fn load<const TRAIN: bool>(dir: &Path) -> Result<Self, String> {
        let prefix = if TRAIN { "train" } else { "t10k" };
    }
}
