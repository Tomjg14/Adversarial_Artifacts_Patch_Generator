CONFIG = {
    # --- Main settings ---
    "mode": "targeted",
    "target_class_id": 859,  # Toaster
    "forbidden_ids": [834, 835, 836, 837, 838, 451, 452, 643, 644, 903, 904, 457, 458],

    # --- Training parameters ---
    "learning_rate": 0.4,
    "epochs": 400,
    "patch_initial_value": 0.5,

    # --- Loss weights ---
    "penalty_weight": 0.7,
    "tv_weight": 0.0,  # Total Variation loss weight to encourage smoothness

    # --- Patch and model dimensions ---
    "original_width": 1280,
    "original_height": 640,
    "model_input_size": 224,
    "imagenet_mean": [0.485, 0.456, 0.406],
    "imagenet_std": [0.229, 0.224, 0.225],

    # --- Expectation-Over-Transformation (EOT) settings ---
    "eot_rotation_degrees": 3,
    "eot_crop_scale": [0.95, 1.05],
    "eot_crop_ratio": [1.0, 1.0],
    "eot_color_jitter_brightness": 0.1,
    "eot_color_jitter_contrast": 0.1,

    # --- Logging and Saving ---
    "log_frequency_epochs": 20,
    "save_examples_last_n_epochs": 100,
    "save_examples_frequency_epochs": 20,
    "training_examples_dir": "training_examples",
    "final_patch_dir": "patches",
}