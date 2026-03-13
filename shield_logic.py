import torch
import torch.nn as nn
from typing import List, Tuple

class PGD:
    """
    Implementation of Projected Gradient Descent (PGD) attack.
    Reference: https://arxiv.org/abs/1706.06083
    """
    def __init__(self, model: nn.Module, eps: float = 0.03, alpha: float = 0.01, steps: int = 10):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_fn = nn.CrossEntropyLoss()

    def perturb(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images_orig = images.clone().detach()
        images = images.clone().detach().to(labels.device)
        
        # Random start
        images = images + torch.empty_like(images).uniform_(-self.eps, self.eps)
        images = torch.clamp(images, 0, 1)

        for _ in range(self.steps):
            images.requires_grad = True
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)
            
            self.model.zero_grad()
            loss.backward()
            
            adv_images = images + self.alpha * images.grad.data.sign()
            eta = torch.clamp(adv_images - images_orig, min=-self.eps, max=self.eps)
            images = torch.clamp(images_orig + eta, min=0, max=1).detach_()
            
        return images

class RobustTrainer:
    """
    A training wrapper for Adversarial Training.
    """
    def __init__(self, model: nn.Module, attacker: PGD):
        self.model = model
        self.attacker = attacker
        self.loss_fn = nn.CrossEntropyLoss()

    def train_step(self, images: torch.Tensor, labels: torch.Tensor, optimizer: torch.optim.Optimizer) -> float:
        # 1. Generate adversarial examples for the batch
        self.model.eval()
        adv_images = self.attacker.perturb(images, labels)
        self.model.train()
        
        # 2. Compute loss on the adversarial examples
        optimizer.zero_grad()
        outputs = self.model(adv_images)
        loss = self.loss_fn(outputs, labels)
        
        # 3. Optimization step
        loss.backward()
        optimizer.step()
        
        return loss.item()

if __name__ == "__main__":
    print("Shield-AI Logic initialized. Ready for robust training simulations.")