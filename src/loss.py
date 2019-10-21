from torch import Tensor, nn
from torch.nn.modules.loss import _Loss
from pytorch_toolbelt.losses.focal import FocalLoss


class CCE(_Loss):
    """Implementation of CCE for 2D model from logits."""

    def __init__(self, from_logits=True, ignore_index=-1):
        super(CCE, self).__init__()
        self.from_logits = from_logits

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


class Focal(_Loss):
    """Implementation of Focal for 2D model from logits."""

    def __init__(self, from_logits=True, alpha=0.5, gamma=2, ignore_index=-1):
        super(Focal, self).__init__()
        self.from_logits = from_logits
        self.ignore_index = ignore_index
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        """

        Args:
            y_pred: NxCxHxW
            y_true: NxHxW

        Returns:

        """
        y_pred = nn.LogSoftmax(dim=1)(y_pred)

        loss = FocalLoss(alpha=self.alpha, gamma=self.gamma, ignore_index=self.ignore_index)

        return loss(y_pred, y_true)
