from time import time
from typing import *

import torch
from loguru import logger as logging
from safetensors import torch as st
from torch import Tensor, nn

from acc23 import *
from acc23.explain import shap
from acc23.models import *
from acc23.models.base_mlc import BaseMultilabelClassifier

DEVICE = "cuda:1"
MODEL_CSL: Type[BaseMultilabelClassifier] = Orchid
CHECKPOINT = "out/tb_logs/orchid/version_21/checkpoints/epoch=25-step=130.ckpt"


class BlindModel(nn.Module):
    model: BaseMultilabelClassifier

    def __init__(self, model: BaseMultilabelClassifier):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor, *_, **__) -> Tensor:
        img = torch.zeros(x.shape[0], N_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
        d = dict(zip(df.columns, x.T))
        y = self.model(d, img).sigmoid()
        y = y.detach().cpu()
        return y


logging.info("Loading model '{}'", CHECKPOINT)
model = MODEL_CSL.load_from_checkpoint(CHECKPOINT)
model.to(DEVICE)
blind_model = BlindModel(model)

logging.info("Loading dataset")
dm = ACCDataModule()
dm.prepare_data()
dm.setup("test")
ds = dm.ds_test
df = ds.data.drop(columns=TARGETS + ["Chip_Image_Name"])
data = torch.tensor(df.to_numpy(), dtype=float)

start = time()
logging.info("Computing SHAP")
s = shap(blind_model, data, batch_size=64)
s = s.contiguous()
st.save_file({"shap": s}, "test.shap.st")
logging.info("Done after {}", time() - start)
