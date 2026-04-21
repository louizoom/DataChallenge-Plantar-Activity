"""
Helper utilities shared across training scripts.

Classes
-------
EarlyStopping : stops training when validation loss stops improving.
"""


class EarlyStopping:
    """Stop training when validation loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 5, min_delta: float = 0.0) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
