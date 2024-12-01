import torch


def soft_dice_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    intersection = (prediction * target).sum(dim=[-2, -1])
    union = torch.square(prediction).sum(dim=[-2, -1]) + torch.square(target).sum(
        dim=[-2, -1]
    )
    return 1.0 - (2.0 * intersection) / (union + epsilon)
