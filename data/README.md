# Wiring data

This package (`data/`) reads dataset locations from a YAML registry. To point
the code at your own dataset copies, edit one file: `data/registry.yaml`.

## Quick start

1. Copy `.env.example` → `.env` and set `DATASETS_ROOT` to the root under
   which your datasets live (defaults to `./datasets/`).
2. Create `data/registry.yaml` (gitignored — local to your machine). Each
   top-level key is a dataset name; under it, set the paths the corresponding
   datamodule expects.

```yaml
# data/registry.yaml
r4d_bench:
  media_dir: PATH_TO/R4D-Bench
  eval_path: PATH_TO/R4D-Bench/test.json
```

The datamodule at [r4d_bench/datamodule.py](r4d_bench/datamodule.py) reads
these keys via `DATASETS[dirname]` (see [config.py](config.py)).

## Adding a new eval dataset

Each datamodule is a class subclassing `BaseDatamodule` and decorated with
`@register_datamodule()`. The class attribute `dirname` is the registry key:

```python
# data/my_bench/datamodule.py
from ..register import register_datamodule
from ..base import BaseDatamodule
from ..config import DATASETS

@register_datamodule()
class MyBench(BaseDatamodule):
    name = "MyBench"
    dirname = "my_bench"            # <-- key in registry.yaml
    registry = DATASETS[dirname]
    eval_path = registry["eval_path"]
    media_dir = registry["media_dir"]
```

Then add to `data/registry.yaml`:

```yaml
my_bench:
  media_dir: /path/to/MyBench/videos
  eval_path: /path/to/MyBench/test.json
```

And import it from [`data/__init__.py`](__init__.py) so the decorator runs:

```python
from .my_bench import *
```

## SFT training mixtures (separate registry)

Training mixtures are resolved by `llava/data/builder.py`, which reads
**`llava/data/registry/datasets/<cluster>.yaml`** + `mixtures.yaml`. This is
a different registry from the one above (used by upstream NVILA tooling).

For 4D-RGPT SFT, [`scripts/nvila/sft.sh`](../scripts/nvila/sft.sh) passes
`--data_mixture sat+vstibench+robofac+wolf` — each name resolves to an entry
under `llava/data/registry/datasets/<cluster>.yaml` with `data_path` and
`media_dir`. The cluster-specific yamls are gitignored; copy
`llava/data/registry/datasets/default.yaml` and add entries like:

```yaml
sat:
    _target_: llava.data.LLaVADataset
    data_path: /path/to/SAT/sft.json
    media_dir: /path/to/SAT
    is_video: false
vstibench:
    _target_: llava.data.LLaVA4DVideoDataset
    data_path: /path/to/VSTI-Bench/train.json
    media_dir: /path/to/VSTI-Bench/video
```

## Currently registered eval datamodules

| Dirname (registry key) | Class | Source dataset |
|---|---|---|
| `r4d_bench` | [r4d_bench/datamodule.py](r4d_bench/datamodule.py) | nvidia/R4D-Bench (HF) |
