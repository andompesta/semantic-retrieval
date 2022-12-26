from torch import nn, Tensor


class ContrastiveLoss(nn.Module):
    """
    InfoNCE loss for multi-model representation learning.

    reduction (str, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
        ``'mean'``: the sum of the output will be divided by the number of
        elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
        and :attr:`reduce` are in the process of being deprecated, and in the meantime,
        specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
    """

    def __init__(
        self,
        reduction: str = 'none',
        alpha: float = 0.5,
    ) -> None:
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha

        self.img_loss = nn.CrossEntropyLoss(reduction=self.reduction)
        self.text_loss = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(
        self,
        logits_per_image: Tensor,
        logits_per_text: Tensor,
        targets: Tensor,
    ) -> Tensor:
        loss_i = self.img_loss(
            logits_per_image,
            targets,
        )

        loss_t = self.text_loss(
            logits_per_text,
            targets,
        )

        return (self.alpha * loss_i) + (loss_t * (1 - self.alpha))
