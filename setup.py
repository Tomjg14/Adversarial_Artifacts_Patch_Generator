from setuptools import setup, find_packages

setup(
    name="adv-patch-generator",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "opencv-python",
        "matplotlib",
        "Pillow",
    ],
    package_data={
        'patch_generator': ['original_glasses/bril_mask.png', 'dataset/*.jpg'],
    },
    entry_points={
        "console_scripts": [
            "generate-patch=patch_generator.patch_generator:generate_patch",
        ],
    },
    python_requires='>=3.6',
)