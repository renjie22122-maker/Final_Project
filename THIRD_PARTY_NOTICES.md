# Third-party code

This repository integrates research implementations; it does not claim authorship of their original methods.

| Local directory | Upstream project | Preserved licence |
| --- | --- | --- |
| `src/cyclegan` | [junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) | `src/cyclegan/LICENSE` |
| `src/cut` | [taesungp/contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) | `src/cut/LICENSE` |
| `src/cyclegan_turbo` | [GaParmar/img2img-turbo](https://github.com/GaParmar/img2img-turbo) | `src/cyclegan_turbo/LICENSE` |
| `src/unsb` | [cyclomon/UNSB](https://github.com/cyclomon/UNSB) | `src/unsb/LICENSE` |

Nested components also retain their own notices, including `src/unsb/models/score_sde/`. The local copies contain modifications and their exact original upstream commit identifiers were not recorded; do not imply that they are pristine or pinned upstream releases.

Stable Diffusion Turbo, DINO, LPIPS/VGG and other pretrained weights are separate dependencies. Their model/data terms apply independently of source-code licences. Original report sample images are retained as research illustrations, not offered as an unrestricted training dataset. No new blanket licence is assigned by this repository curation.
