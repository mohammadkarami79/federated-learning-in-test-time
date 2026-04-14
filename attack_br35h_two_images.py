import os
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt


def load_image(path: str, img_size: int = 224) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),  # [0,1]
    ])
    image = Image.open(path).convert('RGB')
    return transform(image)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.015,
    step_size: float = 0.003,
    steps: int = 7,
) -> torch.Tensor:
    """
    Untargeted PGD in pixel space [0,1], matching log11.txt params.
    Gradients are computed on the normalized inputs expected by the model.
    """
    model.eval()
    adv = images.clone().detach()
    delta = torch.zeros_like(adv).uniform_(-epsilon, epsilon)
    adv = torch.clamp(adv + delta, 0.0, 1.0)

    for _ in range(steps):
        adv.requires_grad_(True)
        logits = model(normalize_imagenet(adv))
        loss = nn.CrossEntropyLoss()(logits, labels)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad = adv.grad.sign()
            adv = adv + step_size * grad
            adv = torch.max(torch.min(adv, images + epsilon), images - epsilon)
            adv = torch.clamp(adv, 0.0, 1.0)
    return adv.detach()


def predict_labels(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = model(normalize_imagenet(images))
        labels = logits.argmax(dim=1)
    return labels


def save_side_by_side(clean: torch.Tensor, adv: torch.Tensor, out_path: str, title: str):
    clean_np = clean.permute(1, 2, 0).cpu().numpy()
    adv_np = adv.permute(1, 2, 0).cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(clean_np)
    axes[0].set_title('Clean')
    axes[0].axis('off')
    axes[1].imshow(adv_np)
    axes[1].set_title('Adversarial')
    axes[1].axis('off')
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(
    img_paths: Tuple[str, str],
    output_dir: str,
    epsilon: float = 0.015,
    step_size: float = 0.003,
    steps: int = 7,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Use ResNet18; gradients are sufficient even without pretrained weights
    model = models.resnet18(weights=None)
    model = model.to(device)
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for img_path in img_paths:
        img_tensor = load_image(img_path).unsqueeze(0).to(device)
        labels = predict_labels(model, img_tensor)
        adv = pgd_attack(model, img_tensor, labels, epsilon, step_size, steps)

        fname = os.path.splitext(os.path.basename(img_path))[0]
        out_file = os.path.join(output_dir, f"{fname}_clean_vs_adv_log11.png")
        save_side_by_side(img_tensor[0].cpu(), adv[0].cpu(), out_file, f"PGD eps={epsilon}, alpha={step_size}, steps={steps}")


if __name__ == '__main__':
    base_folder = r"D:\federeated_learning_in_test_time\federated-learning-in-test-time\data\Br35H\Br35H-Mask-RCNN\TEST"
    img1 = os.path.join(base_folder, 'y703.jpg')
    img2 = os.path.join(base_folder, 'y732.jpg')
    out_dir = os.path.join(base_folder, 'adv_outputs')
    run((img1, img2), out_dir)


