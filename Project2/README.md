# CIFAR-10 Image Classifier

This project trains a convolutional neural network in PyTorch to classify images
into the 10 CIFAR-10 categories: plane, car, bird, cat, deer, dog, frog, horse,
ship, and truck. After training, it evaluates the model and predicts the classes
of four image files.

## Requirements

- Python
- PyTorch and torchvision
- NumPy
- OpenCV (`cv2`)

## Local CIFAR-10 data

The script uses the CIFAR-10 files already on disk and does not download the
dataset. The expected directory layout is:

```text
Project2/
|-- main.py
|-- data/
    |-- cifar-10-batches-py/
        |-- cifar-10-batches-py/
            |-- batches.meta
            |-- data_batch_1
            |-- data_batch_2
            |-- data_batch_3
            |-- data_batch_4
            |-- data_batch_5
            |-- test_batch
```

The inner `cifar-10-batches-py` folder contains the extracted dataset files.
`main.py` locates the data relative to its own location and configures both
training and test datasets with `download=False`. If these extracted files are
missing or incomplete, the dataset loader reports an error instead of fetching
the dataset remotely.

## Run

From the `Project2` directory, run:

```bash
python main.py
```

The script automatically selects CUDA when available and otherwise uses the
CPU. It trains for 25 epochs with batches of 128, prints training and test-set
metrics, and saves the trained model weights as `image_classifier.pth` in the
current working directory.

For the final predictions, place `deer.jpg`, `car.jpg`, `horse.jpg`, and
`plane.jpg` in the current working directory. The script resizes each image to
32-by-32 pixels and prints its predicted CIFAR-10 class. If an image cannot be
read, it reports that and continues with the remaining images.

## Previous Git versions

Earlier versions of `main.py` are available in the repository's Git history.
They used TensorFlow/Keras rather than PyTorch and had a simpler convolutional
network:

- `6d9bb85` was an MNIST experiment: it loaded and displayed training examples
  labeled with their digit classes. It did not define or train a model, and it
  did not explicitly force CPU execution.
- `6d0612d` introduced CIFAR-10 loading and image visualization with
  TensorFlow/Keras.
- `c8dc440` added a basic CNN and trained it for 10 epochs on CIFAR-10.
- `787fce1` retained the 10-epoch CNN while adding image prediction and
  preprocessing/error handling.
- `3bedaef` increased training to 20 epochs, added `Dropout(0.5)`, and added
  prediction for four images.

The TensorFlow CNN versions did not explicitly select or move work to a CUDA
device. They used TensorFlow's default device placement, which could use an
available GPU automatically; they were not explicitly restricted to CPU. The
MNIST experiment likewise did not set a device, and it only visualized samples.
The current version instead selects CUDA when PyTorch reports it is available,
falls back to CPU otherwise, and trains for 25 epochs.

To inspect an older script version, run this from the repository root:

```bash
git show 3bedaef:Project2/main.py
```

Replace `3bedaef` with another commit ID above to inspect that version, for
example `git show 6d9bb85:Project2/main.py` for the MNIST experiment.
