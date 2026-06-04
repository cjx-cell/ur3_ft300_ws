---
license: apache-2.0
library_name: lerobot
pipeline_tag: robotics
---

## Pi0 pretrained model

This repository contains the model described in [π_0: A Vision-Language-Action Flow Model for General Robot Control](https://huggingface.co/papers/2410.24164).

See the [Twitter thread](https://x.com/RemiCadene/status/1886823939856589296) and [blog post](https://huggingface.co/blog/pi0) for more info regarding its integration in [LeRobot](https://github.com/huggingface/lerobot).

## Usage

You can download and use this model with:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
action = policy.select_action(batch)
```

## Fine-tuning

You can easily finetune it on your dataset. For instance on @dana_55517 's [dataset](https://huggingface.co/spaces/lerobot/visualize_dataset?dataset=danaaubakirova%2Fkoch_test&episode=0):
```python
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Take a look at the [code](https://github.com/huggingface/lerobot/blob/main/lerobot/common/policies/pi0/modeling_pi0.py) regarding the implementation.
