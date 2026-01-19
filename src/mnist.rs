use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use flate2::bufread::GzDecoder;

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

        let images_file = dir.join(format!("{prefix}-images-idx3-ubyte.gz"));
        let labels_file = dir.join(format!("{prefix}-labels-idx1-ubyte.gz"));

        let images = load_images(&images_file)?;
        let labels = load_labels(&labels_file)?;

        let num_samples = labels.shape[0];

        Ok(MnistDataset {
            images,
            labels,
            num_samples,
        })
    }
}

fn load_images(path: &PathBuf) -> Result<Tensor, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open {path:?}: {e}"))?;

    let mut reader: Box<dyn Read> = if path.extension().is_some_and(|ext| ext == "gz") {
        Box::new(GzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let magic = read_u32_be(&mut reader)?;
    if magic != 2051 {
        return Err(format!("Invalid magic number for images: {}", magic));
    }

    let num_images = read_u32_be(&mut reader)? as usize;
    let rows = read_u32_be(&mut reader)? as usize;
    let cols = read_u32_be(&mut reader)? as usize;

    if rows & cols != 28 {
        return Err(format!("Unexpected image dimensiones: {rows}x{cols}"));
    }

    let mut data = vec![0u8; num_images * rows * cols];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read image data: {e}"))?;
    let float_data: Vec<f32> = data.iter().map(|&x| x as f32 / 255.0).collect();

    Ok(Tensor {
        data: float_data,
        shape: vec![num_images, 1, 28, 28],
    })
}

fn load_labels(path: &PathBuf) -> Result<Tensor, String> {
    let file = File::open(path).map_err(|e| format!("Failed to open {path:?}: {e}"))?;

    let mut reader: Box<dyn Read> = if path.extension().map_or(false, |ext| ext == "gz") {
        Box::new(GzDecoder::new(BufReader::new(file)))
    } else {
        Box::new(BufReader::new(file))
    };

    let magic = read_u32_be(&mut reader)?;
    if magic != 2049 {
        return Err(format!("Invalid magic number for labels {magic}"));
    }

    let num_labels = read_u32_be(&mut reader)? as usize;
    let mut data = vec![0u8; num_labels];
    reader
        .read_exact(&mut data)
        .map_err(|e| format!("Failed to read image data: {e}"))?;
    let float_data: Vec<f32> = data.iter().map(|&label| label as f32).collect();

    Ok(Tensor {
        data: float_data,
        shape: vec![num_labels],
    })
}

fn read_u32_be(reader: &mut dyn Read) -> Result<u32, String> {
    let mut buf = [0u8; 4];
    reader
        .read_exact(&mut buf)
        .map_err(|e| format!("Failed to read u32: {}", e))?;
    Ok(u32::from_be_bytes(buf))
}

pub fn download_instructions() -> &'static str {
    r#"
MNIST dataset not found. Please run the following command:
    mkdir data && cd data
    wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
    wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
    wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
    wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
"#
}
