from torch import Tensor, nn
from torch.nn.modules.loss import _Loss


class CCE(_Loss):
    """Implementation of CCE for 2D model from logits."""

    def __init__(self, from_logits=True, weight=None, smooth=1e-7, ignore_index=-1):
        super(CCE, self).__init__()
        self.from_logits = from_logits
        self.weight = weight
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        Args:
            y_pred: NxCxHxW
            y_true: NxHxW

        Returns:

        """
        y_pred = nn.LogSoftmax(dim=1)(y_pred)

        loss = nn.NLLLoss(ignore_index=self.ignore_index)

        return loss(y_pred, y_true)
