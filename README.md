# GCA-GDA: Geometry-Consistent Classifier Adjustment for Graph Domain Adaptation

Anonymous implementation of **GCA-GDA** for unsupervised graph domain adaptation.

## Setup

Requires Python 3.10+ and PyTorch. Install the remaining dependencies with:

```bash
pip install numpy scipy pandas networkx scikit-learn
```

For MAG graphs serialized as PyTorch-Geometric objects, also install `torch-geometric` with versions compatible with your PyTorch environment.

## Data

Datasets are **not included**. Set `--data-dir` to the benchmark data directory. Supported families are Citation (ArnetMiner), Blog (Social Networks), Twitch, MAG, and Elliptic, covering **24 transfer tasks**. Expected inputs are `.mat` (Citation/Blog), CSV/JSON (Twitch), `.pt` (MAG), and `.pkl` (Elliptic), following the paths in `gca_gda/data.py`. Load serialized datasets only from trusted sources.

## Run

List available transfer tasks:

```bash
python main.py --family all --list-tasks
```

Run Citation C → D with five fixed seeds:

```bash
python main.py --family citation --source citationv1 --target dblpv7 \
  --data-dir data --seeds 100 101 102 103 104 \
  --lambda-j 0.1 --scale 24 --tau 1.0 --epsilon 0.1 \
  --evaluate-target
```

Run all 24 tasks:

```bash
python main.py --family all --all-tasks --data-dir data \
  --seeds 100 101 102 103 104 --evaluate-target
```

Training defaults: **400 epochs**, **100-epoch joint warm-up**, **40-epoch contrastive ramp**, and classifier refinement every **5 epochs**. Use CLI flags or `--config path/to/config.json` to override settings. Results are written to `outputs/`.
