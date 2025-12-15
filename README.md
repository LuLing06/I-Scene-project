# I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners

<p align="center">
  🏠 <a href="https://luling06.github.io/I-Scene-project/"><b>Project Page</b></a> | 
  📄 <a href="#"><b>Paper</b></a> | 
  🤗 <a href="#"><b>Model</b></a> | 
  📦 <a href="#"><b>Dataset</b></a> | 
  🎮 <a href="#"><b>Online Demo</b></a>
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
  <img src="web_html/teaser.png" alt="Teaser" width="800">
</p>

## 🔑 Key Features

- **Model Flexibility:** A pre-trained 3D instance generator can be directly reprogrammed as a scene-level spatial learner, without scene-level annotations.
- **Transferable Spatial Prior:** The reprogrammed model's spatial prior provides a rich learning signal for inferring proximity, support, and symmetry from purely geometric cues.
- **Data Independence:** The model learns spatial knowledge on non-semantic scenes from randomly composed objects, removing dependency on annotated data.
- **Strong Generalizability:** It allows for easy generalization to unseen layouts and various spatial relations in a feed-forward manner without per-scene optimization.

## 🔥 Updates

- [ ] Release inference code
- [ ] Release sparse structure latent VAE weights
- [ ] Release interactive huggingface demo and usage
- [ ] Release training data
- [ ] Release evaluation code


## 🚀 Demo

Visit our **[Project Page](https://luling06.github.io/I-Scene-project/)** for:
- Interactive 3D scene visualization
- Comparison with state-of-the-art methods
- More visualization examples

## 📦 Installation

```bash
# Coming soon
git clone https://github.com/LuLing06/I-Scene-project.git
cd I-Scene-project
```

## 🎯 Usage

```bash
# Coming soon
```


## 📜 Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{ling2025iscene,
  title={I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners},
  author={Ling, Lu and Ge, Yunhao and Sheng, Yichen and Bera, Aniket},
  journal={arXiv preprint arXiv:2506.XXXXX},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

We thank the authors of [TRELLIS](https://github.com/microsoft/TRELLIS), and other related works for their inspiring research.

---

<p align="center">
  <i>The website template is borrowed from <a href="https://nerfies.github.io/">Nerfies</a> and <a href="https://github.com/VAST-AI-Research/MIDI-3D">MIDI</a>.</i>
</p>

