# MNIST-CNN
MNIST digit recognition solver using Convolutional Neural Networks

## Run

If you just want to quickstart the program run:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
mkdir data && cd data
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz
wget https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz
./target/release/mnist-cnn
```

Which will:
- Create data/ directory containing the mnist dataset necessary files
- Compile the program using native cpu instructions for maximum performance
- Run it from the build directory which in cargo is './target/[BUILD-MODE]/[EXE-NAME]'
