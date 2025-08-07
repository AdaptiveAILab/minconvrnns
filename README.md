## Minimal Convolutional RNNs Accelerate Spatiotemporal Learning<br><sub>Official PyTorch Implementation</sub>

This repository contains the PyTorch implementation featuring code for data generation, model training, and evaluation of our paper.
> 📄 [Minimal Convolutional RNNs Accelerate Spatiotemporal Learning](https://arxiv.org/abs/2508.03614)<br>
> [Coşku Can Horuz](https://www.linkedin.com/in/cosku-horuz/), [Sebastian Otte](https://www.linkedin.com/in/sebastian-otte/), [Martin V. Butz](https://www.linkedin.com/in/martin-butz-85b971150/), [Matthias Karlbauer](https://www.linkedin.com/in/matthias-karlbauer/)\
> <br> [Adaptive AI Lab](https://adaptiveailab.github.io/), University of Lübeck
> <br> [Neuro-Cognitive Modeling Group](https://uni-tuebingen.de/fakultaeten/mathematisch-naturwissenschaftliche-fakultaet/fachbereiche/informatik/lehrstuehle/cognitive-modeling/), University of Tübingen<br>

Our work introduces efficient, minimal and parallel scan compatible convolutional RNN architectures for spatiotemporal prediction tasks. These lightweight models demonstrate strong performance on synthetic and real-world datasets while being significantly faster and simpler than existing baselines.

---

## 📦 Getting Started

To install the `mcrnn` (Minimal Convolutional RNN) package:

1. Create and activate a new environment:

   ```bash
   conda create -n mcrnn python=3.11 -y && conda activate mcrnn
   ```
2. Change into the `mcrnn` directory:

   ```bash
   cd src/mcrnn
   ```
3. Install the package:

   ```bash
   pip install -e .
   ```

---

## 📁 Data

### 🌀 Navier-Stokes

> ⚠️ Data generation requires **PyTorch v1.6** — we recommend using a separate environment to avoid version conflicts.

1. Create a separate environment:

   ```bash
   conda create -n nsgen python=3.7 -y && conda activate nsgen
   pip install torch==1.6.0 "xarray[parallel]" scipy einops tqdm matplotlib
   ```
2. Generate data:

   ```bash
   python data/data_generation/navier-stokes/generate_ns_2d.py
   ```
3. Data geneneration can take long, depending on the number of samples that are generated and the simulation configuration. A small data set for experimentation with 5 samples and a simulation time of 50 can be created as follows

   ```bash
   python data/data_generation/navier-stokes/generate_ns_2d.py -n 5 -t 50 -b 1
   ```

To create train/val/test sets:

```bash
python data/data_generation/navier-stokes/generate_ns_2d.py -n 1000 -t 50  # train
python data/data_generation/navier-stokes/generate_ns_2d.py -n 50 -t 50    # val
python data/data_generation/navier-stokes/generate_ns_2d.py -n 200 -t 51   # test
```

> [!TIP]
Visualize a single sample in an animation

```bash
python data/data_generation/navier-stokes/generate_ns_2d.py --animate -n 1 -b 1 -r 32
```

### 🌍 Geopotential (Real-World Data)

As a real-world case, we consider the air pressure field at 500hPa as a key component of our planet's weather system. [WeatherBench](https://github.com/pangeo-data/WeatherBench) offers various atmospheric variables on different spatial resolutions. We use $`\Phi_{500}`$ (geopotential at 500hPa) at a global spatial resolution of 5.625° from 1979-2018 on a temporal resolution of one hour. A single time step has the shape `[1, 32, 64]` in `[C, H, W]`.

Run the following commands to download the data (compressed ~1.4GB) and extract it to the appropriate directory in the repository.

> ```bash
> mkdir -p data/netcdf/
> rsync -P rsync://m1524895@dataserv.ub.tum.de/m1524895/5.625deg/geopotential_500/geopotential_500_5.625deg.zip data/netcdf/geopotential_500_5.625deg.zip
> ```
> enter password `m1524895`
> ```bash
> unzip data/netcdf/geopotential_500_5.625deg.zip -d data/netcdf/geopotential_500_5.625deg
> rm data/netcdf/geopotential_500_5.625deg.zip
> ```

---

## 🏋️‍♀️ Training

Activate the environment:

```bash
conda activate mcrnn
```

Train a model (e.g. 2-layer ConvLSTM with 4 cells each, CPU):

```bash
python scripts/train.py model=convlstm model.hidden_sizes=[4,4] model.name=my_clstm4-4_model data=navier-stokes device="cpu"
```

Train on GPU with default 16 cells:

```bash
python scripts/train.py model=convlstm model.name=my_clstm16-16_model data=navier-stokes device=cuda:0
```

More examples and all training commands used in this study are listed in the bash files:

* [`train_ns_s15_tf5.sh`](src/mcrnn/train_ns_s15_tf5.sh)
* [`train_ns_s25_tf20.sh`](src/mcrnn/train_ns_s25_tf20.sh)
* [`train_geopotential.sh`](src/mcrnn/train_geopotential.sh)

> [!TIP]
Use TensorBoard to inspect training:

```bash
tensorboard --logdir outputs
# Open http://localhost:6006/
```

---

## 📈 Evaluation

To evaluate a trained model, run

```bash
python scripts/evaluate.py -c outputs/my_clstm4-4_model
```

The correct model.name must be provided as -c argument (standing for "checkpoint"). The evaluation script will compute an RMSE and animate the predicted dynamics next to the ground truth.

Multi- and cross-model evaluations can be performed by passing multiple model names, e.g.,

```bash
python scripts/evaluate.py -c outputs/my_clstm4-4_model outputs/my_clstm16-16_model
```

Wildcards can be used to indicate a family of models by name, e.g.,

```bash
python scripts/evaluate.py -c outputs/*clstm*
```
evaluates all models in the `outputs` directory that have `clstm` in their name.

---

## 📖 Citation

If you find this repository helpful, please cite:

```bibtex
@article{horuz2025minimal,
  title={Minimal Convolutional RNNs Accelerate Spatiotemporal Learning},
  author={Horuz, Co\c{s}ku Can and Otte, Sebastian and Butz, Martin V. and Karlbauer, Matthias},
  journal={arXiv preprint arXiv:2508.03614},
  year={2025}
}
```
---
## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is licensed under the MIT License.
