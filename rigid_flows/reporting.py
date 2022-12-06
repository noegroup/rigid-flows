from jax import Array
from jax_dataclasses import pytree_dataclass
from tensorboardX import SummaryWriter


@pytree_dataclass(frozen=True)
class Reporter:
    writer: SummaryWriter
    prefix: tuple[str, ...]

    def write_text(self, name: str, text: str, *args, **kwargs):
        path = self.prefix + (name,)
        self.writer.add_text("/".join(path), text, *args, **kwargs)

    def write_scalar(self, name: str, scalar: Array | float, *args, **kwargs):
        path = self.prefix + (name,)
        self.writer.add_scalar("/".join(path), scalar, *args, **kwargs)

    def write_hparams(self, hparams: dict, metric_dict: dict, *args, **kwargs):
        self.writer.add_hparams(hparams, metric_dict, *args, **kwargs)
