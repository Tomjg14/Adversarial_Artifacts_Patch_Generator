import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import glob

from .config import CONFIG


class PatchGenerator:
    """
    Encapsulates the logic for generating an adversarial patch.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = self._get_device()
        self._DIR = os.path.dirname(os.path.abspath(__file__))

        # Load model and set up transforms
        self.model = self._load_model()
        self.normalize = transforms.Normalize(mean=self.config['imagenet_mean'], std=self.config['imagenet_std'])
        self.eot_transforms = self._get_eot_transforms()

        # Load data
        self.faces = self._get_faces()
        self.mask_large = self._get_mask()
        self.clean_class = self._get_clean_class()

        # Initialize patch
        self.patch_high_res = torch.full(
            (3, self.config['original_height'], self.config['original_width']),
            self.config['patch_initial_value'],
            device=self.device,
            requires_grad=True
        )
        self.optimizer = torch.optim.Adam([self.patch_high_res], lr=self.config['learning_rate'])

    def _get_device(self) -> torch.device:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _load_model(self):
        model = models.resnet50(weights='DEFAULT').to(self.device).eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _get_eot_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomRotation(degrees=self.config['eot_rotation_degrees']),
            transforms.RandomResizedCrop(
                self.config['model_input_size'],
                scale=tuple(self.config['eot_crop_scale']),
                ratio=tuple(self.config['eot_crop_ratio']),
                antialias=True
            ),
            transforms.ColorJitter(
                brightness=self.config['eot_color_jitter_brightness'],
                contrast=self.config['eot_color_jitter_contrast']
            )
        ])

    def _get_faces(self) -> list:
        input_size = (self.config['model_input_size'], self.config['model_input_size'])
        to_tensor_224 = transforms.Compose([transforms.Resize(input_size), transforms.ToTensor()])
        faces = []
        dataset_path = os.path.join(self._DIR, "dataset", "*.jpg")
        face_paths = sorted(glob.glob(dataset_path))
        if not face_paths:
            raise FileNotFoundError(f"No faces found matching {dataset_path}")

        for f_path in face_paths:
            faces.append(to_tensor_224(Image.open(f_path).convert("RGB")).to(self.device))
        return faces

    def _get_mask(self) -> torch.Tensor:
        mask_path = os.path.join(self._DIR, "original_glasses", "bril_mask.png")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found at {mask_path}")
        return transforms.ToTensor()(Image.open(mask_path).convert("RGB")).to(self.device)

    def _get_clean_class(self) -> int:
        with torch.no_grad():
            clean_output = self.model(self.normalize(self.faces[0]).unsqueeze(0))
            return torch.argmax(clean_output, dim=1).item()

    def _save_training_example(self, image_tensor: torch.Tensor, epoch: int):
        """Saves a single training example image with the patch applied."""
        output_dir = self.config['training_examples_dir']
        os.makedirs(output_dir, exist_ok=True)

        # Detach from graph, move to CPU, and convert to PIL Image
        image_tensor = image_tensor.cpu().detach()
        pil_image = transforms.ToPILImage()(image_tensor)

        filename = f"epoch_{epoch:04d}_example.png"
        filepath = os.path.join(output_dir, filename)
        pil_image.save(filepath)
        print(f"Saved training example to {filepath}")

    def train(self):
        """Runs the training loop to optimize the patch."""
        print("Starting patch generation...")
        for epoch in range(self.config['epochs']):
            self.optimizer.zero_grad()
            applied_patch = torch.sigmoid(self.patch_high_res)

            # Downsample patch and mask for applying to 224x224 images
            input_size = (self.config['model_input_size'], self.config['model_input_size'])
            full_glasses = applied_patch * self.mask_large
            glasses_224 = F.interpolate(full_glasses.unsqueeze(0), size=input_size, mode='area')[0]
            mask_224 = F.interpolate(self.mask_large.unsqueeze(0), size=input_size, mode='area')[0]

            batch_loss = 0
            output = None  # To ensure output is available for logging
            adv_img = None  # To hold the last generated adversarial image for saving
            for face in self.faces:
                adv_img = (face * (1 - mask_224)) + (glasses_224 * mask_224)
                adv_img_transformed = self.eot_transforms(adv_img)
                output = self.model(self.normalize(adv_img_transformed).unsqueeze(0))

                loss_penalty = torch.mean(output[0, self.config['forbidden_ids']])

                if self.config['mode'] == 'targeted':
                    loss_adversarial = -output[0, self.config['target_class_id']]
                else:  # untargeted
                    loss_adversarial = output[0, self.clean_class]

                batch_loss += loss_adversarial + (self.config['penalty_weight'] * loss_penalty)

            # Total Variation Loss
            tv_loss = torch.sum(torch.abs(applied_patch[:, :, :-1] - applied_patch[:, :, 1:])) * self.config.get('tv_weight', 0.0)
            total_loss = (batch_loss / len(self.faces)) + tv_loss
            total_loss.backward()
            self.optimizer.step()

            # Save an example image near the end of training
            save_epoch_start = self.config['epochs'] - self.config['save_examples_last_n_epochs']
            if epoch >= save_epoch_start and epoch % self.config['save_examples_frequency_epochs'] == 0 and adv_img is not None:
                self._save_training_example(adv_img, epoch)

            if epoch % self.config['log_frequency_epochs'] == 0 and output is not None:
                with torch.no_grad():
                    new_class = torch.argmax(output).item()
                    print(f"Epoch {epoch:3d} | AI thinks: {new_class} | Loss: {total_loss.item():.4f}")

    def export(self, output_dir: str = None):
        """Saves the generated patch to a file."""
        if output_dir is None:
            output_dir = self.config['final_patch_dir']

        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            final_patch = torch.sigmoid(self.patch_high_res).cpu()
            final_export = (final_patch * self.mask_large.cpu()) + (1 - self.mask_large.cpu())
            filename = f"{self.config['mode']}_bril_PIXEL_PERFECT.png"
            filepath = os.path.join(output_dir, filename)
            transforms.ToPILImage()(final_export).save(filepath)
            print(f"Klaar! Opgeslagen als {filepath}")


def generate_patch():
    """Initializes and runs the patch generation process."""
    try:
        generator = PatchGenerator(config=CONFIG)
        generator.train()
        generator.export()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset and mask files are correctly placed.")


if __name__ == "__main__":
    generate_patch()
