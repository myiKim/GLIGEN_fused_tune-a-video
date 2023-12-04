# GLIGEN_fused_tune-a-video

This repository merges GLIGEN, providing video generation control, with tune-a-video, a one-shot editing method. Explore the synergy for advanced and unique video content creation.

## Installation

To install the repo, you might want to do the followings:

```bash
git clone https://github.com/your_username/GLIGEN_fused_tune-a-video.git
cd GLIGEN_fused_tune-a-video
pip install -r requirements.txt
```

## Training and Validation

To Finetune(one-shot training) with validation, you might want to do the followings:

```bash
accelerate launch train_tuneavideo.py --config="configs/gligen_init_test.yaml"
```
Please note that I did the training on V100 GPU on Colab.


## Configuation
If you want to test your own prompt, you need to modify the configurations.

## Acknowledgements
My code relies significantly on Tune-A-Video (https://github.com/showlab/Tune-A-Video). I am grateful to the original authors for sharing their code and models, which served as a crucial foundation for this project.
