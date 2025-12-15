# I-Scene: 3D Instance Models are Implicit Generalizable Spatial Learners

<p align="center">
  <a href="https://luling06.github.io/I-Scene-project/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge" alt="Project Page">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-2506.XXXXX-b31b1b?style=for-the-badge" alt="arXiv">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
  </a>
</p>

<p align="center">
  <b>Lu Ling<sup>1</sup></b>, 
  <b>Yunhao Ge<sup>2</sup></b>, 
  <b>Yichen Sheng<sup>2</sup></b>, 
  <b>Aniket Bera<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Purdue University &nbsp;&nbsp; <sup>2</sup>NVIDIA
</p>

---

## 🌟 Overview

**I-Scene** reprograms a pre-trained 3D instance generator to act as a scene-level learner, replacing dataset-bounded supervision with model-centric spatial supervision. This unlocks the generator's transferable spatial knowledge, enabling generalization to unseen layouts and novel object compositions.

<p align="center">
  <img src="web_html/video2/input.png" alt="Teaser" width="600">
</p>

## 📖 Abstract

Generalization remains the central challenge for interactive 3D scene generation. Existing learning-based approaches ground spatial understanding in limited scene datasets, restricting generalization to new layouts. We instead reprogram a pre-trained 3D instance generator to act as a scene-level learner, replacing dataset-bounded supervision with model-centric spatial supervision. This reprogramming unlocks the generator's transferable spatial knowledge, enabling generalization to unseen layouts and novel object compositions.

Remarkably, **spatial reasoning still emerges even when the training scenes are randomly composed objects**. This demonstrates that the generator's transferable scene prior provides a rich learning signal for inferring proximity, support, and symmetry from purely geometric cues.

## 🔑 Key Findings

- **Geometric cues alone** provide a strong learning signal for spatial reasoning, without high-level scene semantics.
- **Non-semantic spatial relations** are a valuable supervision source; scaling non-semantic data provides richer spatial coverage.

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

## 📊 Results

| Training Dataset | 3D-FRONT CD-S↓ | 3D-FRONT F-Score-S↑ | BlendSwap CD-S↓ | BlendSwap F-Score-S↑ |
|------------------|----------------|---------------------|-----------------|----------------------|
| 3D-FT (25K)      | **0.0137**     | **93.77**           | 0.0118          | 90.79                |
| Rand-15K         | 0.0496         | 79.96               | 0.0081          | 92.67                |
| Rand-25K         | 0.0406         | 81.39               | 0.0075          | 93.60                |
| 3D-FT+Rand-15K   | 0.0148         | 93.50               | **0.0059**      | **94.26**            |

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

## 🙏 Acknowledgments

We thank the authors of [MIDI](https://huanngzh.github.io/MIDI-Page/), [TRELLIS](https://github.com/microsoft/TRELLIS), and other related works for their inspiring research.

---

<p align="center">
  <i>The website template is borrowed from <a href="https://nerfies.github.io/">Nerfies</a>.</i>
</p>

