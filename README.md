# I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners

<p align="center">
  🌟 CVPR 2026
</p>

<p align="center">
  🏠 <a href="https://luling06.github.io/I-Scene-web-page/"><b>Project Page</b></a> |
  📄 <a href="https://arxiv.org/abs/2512.13683"><b>Paper</b></a> |
  🤗 <a href="https://huggingface.co/LuLing/IScene"><b>Model</b></a> |
  📦 <a href="#"><b>Dataset</b></a> |
  🎮 <a href="#-demo"><b>Demo</b></a>
</p>

<p align="center">
  <b>Lu Ling<sup>1</sup></b>,
  <b>Yunhao Ge<sup>2</sup></b>,
  <b>Yichen Sheng<sup>2</sup></b>,
  <b>Aniket Bera<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Purdue University &nbsp;&nbsp; <sup>2</sup>NVIDIA Research
</p>

---

## 🌟 Overview

**I-Scene** reprograms a pre-trained 3D instance generator to act as a scene-level learner, replacing dataset-bounded supervision with model-centric spatial supervision. This unlocks the generator's transferable spatial knowledge, enabling generalization to unseen layouts and novel object compositions.

<p align="center">
  <img src="https://raw.githubusercontent.com/LuLing06/I-Scene-web-page/main/web_html/project_teaser.gif" alt="Teaser" width="800">
</p>

## 🔑 Key Features

- **Model Flexibility:** A pre-trained 3D instance generator can be directly reprogrammed as a scene-level spatial learner, without scene-level annotations.
- **Transferable Spatial Prior:** The reprogrammed model's spatial prior provides a rich learning signal for inferring proximity, support, and symmetry from purely geometric cues.
- **Data Independence:** The model learns spatial knowledge on non-semantic scenes from randomly composed objects, removing dependency on annotated data.
- **Strong Generalizability:** It allows for easy generalization to unseen layouts and various spatial relations in a feed-forward manner without per-scene optimization.

## 🔥 Updates

- [x] Release inference code and sparse structure flow transformer
- [x] Release I-Scene-v1 inference checkpoint
- [x] Release local Gradio demo
- [ ] Release hosted online demo, e.g. Hugging Face Space
- [ ] Release training data scripts
- [ ] Release evaluation code

## 📦 Installation

The current release is inference-only. We tested with Python 3.10, CUDA 12.x, PyTorch 2.4.0 + CUDA 12.1, and an NVIDIA H100 GPU.

Quick setup:

```bash
git clone https://github.com/LuLing06/I-Scene-project.git
cd I-Scene-project
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install -e .
pip install -r requirements.txt
pip install spconv-cu120==2.3.6
pip install xformers==0.0.27.post2
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

Notes:

- Install the PyTorch wheel that matches your CUDA version if you are not using CUDA 12.1.
- GLB mesh export also needs `nvdiffrast`, `kaolin`, `xatlas`, `pyvista`, `pymeshfix`, and `igraph`.
- These CUDA/PyTorch extension packages are version-sensitive; install compatible wheels for your machine.

For the interactive demo:

```bash
pip install -r requirements-demo.txt
```

## 🚀 Demo

Visit our **[Project Page](https://luling06.github.io/I-Scene-web-page/)** for:

- Interactive 3D scene visualization
- Comparison with state-of-the-art methods
- More visualization examples

This repository includes a local Gradio demo for interactive segmentation and generation:

```bash
python interactive_demo.py --model LuLing/IScene
```

This is different from a hosted web demo. A hosted online demo, such as a Hugging Face Space, will be released separately.

## 🎯 Inference

The public release loads the I-Scene checkpoint from Hugging Face:

```text
LuLing/IScene
```

Minimal Python example:

```python
from iscene.inference.inferencer import ISceneInferencer

inferencer = ISceneInferencer.from_pretrained("LuLing/IScene")
inferencer.infer_and_save_scene(
    scene_rgb_path="examples/Scenethesis/children_playroom2_rgb.png",
    instance_seg_path="examples/Scenethesis/children_playroom2_seg.png",
    output_dir="outputs/example_gs",
    only_3dgs=True,
    seed=43,
)
```

Run Gaussian Splatting output on a bundled example:

```bash
python run_inference.py \
  --model LuLing/IScene \
  --rgb examples/Scenethesis/children_playroom2_rgb.png \
  --mask examples/Scenethesis/children_playroom2_seg.png \
  --output_dir outputs/example_gs \
  --only_3dgs
```

Run GLB mesh export:

```bash
python run_inference.py \
  --model LuLing/IScene \
  --rgb examples/Scenethesis/children_playroom2_rgb.png \
  --mask examples/Scenethesis/children_playroom2_seg.png \
  --output_dir outputs/example_glb
```

Important options:

```text
--model       I-Scene model id or local model package path. Default: LuLing/IScene
--base_model  Optional TRELLIS base model id or local mirror path.
--seed        Random seed. Default: 43
--simplify    Mesh simplification ratio for GLB export. Default: 0.95
--save_dbg    Save debug artifacts.
--only_3dgs   Skip GLB export and save Gaussian PLY files only.
```

Typical outputs:

```text
scene_pred.ply                 # merged Gaussian Splatting scene
scene_pred.glb                 # merged mesh scene, if GLB export is enabled
instance_XX.*                  # per-instance asset for mask label XX
```

## 🧪 Verification

The current Hugging Face model `LuLing/IScene` was checked against the original implementation for the current paired examples. Stage 1 sparse structure generation matched exactly for all tested pairs: labels, preprocessing tensor hashes, sparse coordinate hashes, and per-slot occupancy.

Stage 2 and final asset export can show small numerical differences on GPU, so the release criterion is exact Stage 1 agreement plus visually consistent final outputs.

## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{ling2025iscene,
  title={I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners},
  author={Ling, Lu and Ge, Yunhao and Sheng, Yichen and Bera, Aniket},
  journal={arXiv preprint arXiv:2512.13683},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

I-Scene builds on the TRELLIS image-to-3D backbone and uses `microsoft/TRELLIS-image-large` as the base model. Please preserve upstream TRELLIS attribution and license terms when redistributing derived code or model packages.

## Acknowledgments

We thank the authors of [TRELLIS](https://github.com/microsoft/TRELLIS), and other related works for their inspiring research.

---

<p align="center">
  <i>The website template is borrowed from <a href="https://nerfies.github.io/">Nerfies</a> and <a href="https://github.com/VAST-AI-Research/MIDI-3D">MIDI</a>.</i>
</p>
