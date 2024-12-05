import torch


def soft_dice_loss(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction shape {prediction.shape} does not match target shape {target.shape}."
        )
    if prediction.device != target.device:
        raise ValueError(
            f"Prediction device {prediction.device} does not match target device {target.device}."
        )
    if prediction.dim() != 4:
        raise ValueError(f"Prediction must be 4D, but got {prediction.dim()}D.")
    if prediction.shape[-1] != 1:
        raise ValueError(
            f"Prediction must have 1 channel, but got {prediction.shape[-1]}."
        )
    intersection = (prediction * target).sum(dim=[-3, -2])
    union = torch.square(prediction).sum(dim=[-3, -2]) + torch.square(target).sum(
        dim=[-3, -2]
    )
    return 1.0 - (2.0 * intersection) / (union + epsilon)
