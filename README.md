# Adversarial Patch Generator

This project generates an adversarial patch designed to look like a pair of glasses. When this patch is applied to an image of a face, it aims to mislead a ResNet50 image classification model into either misclassifying the image (untargeted mode) or classifying it as a specific target class (targeted mode).

The project is built with PyTorch and is structured as an installable Python package with a simple command-line interface.

## Features

- Generates adversarial patches using gradient-based optimization.
- Supports both **targeted** and **untargeted** attack modes.
- Uses Expectation-Over-Transformation (EOT) to create patches that are robust to changes in rotation, scale, and color.
- Fully configurable via a central `config.py` file.
- Packaged for easy installation and use via a `generate-patch` command.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone <your-repository-url>
    cd adversarial_artifacts_patch_generator
    ```

2.  **Install the package:**
    It is recommended to use a virtual environment. Once your environment is activated, install the package in "editable" mode. This will also install all required dependencies from `setup.py`.
    ```bash
    pip install -e .
    ```

## Usage

After installation, you can run the patch generation process from any directory in your terminal:

```bash
generate-patch
```

The script will:

- Start the training process, logging progress to the console.
- Save the final generated patch to the `patches/` directory in your current working folder.
- Save intermediate examples during the final epochs to the `training_examples/` directory.

## Configuration

All settings for the patch generation process are located in `src/patch_generator/config.py`.

You can easily modify parameters such as:

- `mode`: Set to `"targeted"` or `"untargeted"`.
- `target_class_id`: The ImageNet class ID to target in targeted mode.
- `epochs`: The number of training iterations.
- `learning_rate`: The learning rate for the optimizer.
- `eot_*`: Parameters for the Expectation-Over-Transformation augmentations.

## Project Structure

The project requires specific data to be in place:

- **Dataset**: Input face images (e.g., `face1.jpg`, `face2.jpg`) must be placed in `src/patch_generator/dataset/`.
- **Patch Mask**: The mask that defines the shape of the glasses patch (`bril_mask.png`) must be located in `src/patch_generator/original_glasses/`.

## License

This project is licensed under the MIT License.
