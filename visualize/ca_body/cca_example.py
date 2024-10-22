import sys
import torch as th

sys.path.insert(0, 'data')
from attrdict import AttrDict

from omegaconf import OmegaConf

from visualize.ca_body.utils.module_loader import load_from_config
from visualize.ca_body.utils.train import load_checkpoint

device = th.device('cuda:0')

# NOTE: make sure to download the data
model_dir = 'notebooks/checkpoints/ca_body/data/RLW104/'

ckpt_path = f'{model_dir}/body_dec.ckpt'
config_path = f'{model_dir}/config.yml'
assets_path = f'{model_dir}/static_assets.pt'

# config
config = OmegaConf.load(config_path)
# assets
static_assets = AttrDict(th.load(assets_path, weights_only=True))
# sample batch
batch = th.load(f'{model_dir}/sample_batch.pt', weights_only=True)
batch = {
    key: val.to(device) if th.is_tensor(val) else val
    for key, val in batch.items()
}
# batch = to_device(batch, device)
batch.keys()

# building the model
model = load_from_config(
    config.model,
    assets=static_assets,
).to(device)

# loading model checkpoint
load_checkpoint(
    ckpt_path,
    modules={'model': model},
    # NOTE: this is accounting for difference in LBS impl
    ignore_names={'model': ['lbs_fn.*']},
)

# disabling training-only stuff
model.learn_blur_enabled = False
model.pixel_cal_enabled = False
model.cal_enabled = False

# forward
with th.no_grad():
    preds = model(**batch)

    # visualizing
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt

    rgb_preds_grid = make_grid(preds['rgb'], nrow=4).permute(1, 2, 0).cpu().numpy() / 255.
    plt.figure(figsize=(15, 15))
    plt.imshow(rgb_preds_grid[::4, ::4])