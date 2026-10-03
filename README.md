# Neutrino Energy Reconstruction Tutorial

This outreach tutorial shows how a convolutional neural network (CNN) can estimate a neutrino's energy from two simulated detector images. It is written for students who know basic Python but do not need prior particle-physics or machine-learning experience.

The main lesson is not simply that “AI works.” The notebook asks a more scientific question:

> When does the CNN improve the energy estimate, and where does it remain biased?

## Why neutrino energy matters

Neutrinos can change flavor while traveling. In a simplified two-flavor picture, the oscillation probability contains a term like

$$
P \sim \sin^2(2\theta)\,\sin^2\!\left(1.27\,\frac{\Delta m^2 L}{E}\right),
$$

where $L$ is the travel distance and $E$ is the neutrino energy. If the reconstructed energy is wrong, the inferred oscillation pattern can be distorted.

## What does a detector actually see?

The incoming neutrino leaves no visible track before interacting. A simplified electron-neutrino charged-current interaction is

$$
\nu_e + N \rightarrow e^- + X,
$$

where $N$ is a nucleus and $X$ represents the hadronic particles produced in the interaction.

The detector records energy deposited by the outgoing electron and hadrons. These deposits are represented here as two perpendicular 2D views. The CNN receives those two sparse images and predicts one continuous number: the neutrino energy in GeV. This is a **regression** problem, not a classification problem.

## Why reconstruction is difficult

The visible detector activity is not a perfect measurement of the original neutrino energy:

- some energy can be carried by particles that are difficult to detect;
- nuclear breakup and interactions inside the nucleus alter the final state;
- detector thresholds and nonlinear response affect small deposits;
- particle showers can overlap or leave the displayed region;
- different interaction modes can create very different image patterns.

In simulation, the original neutrino energy is known and can be used as the training target. In real detector data, that “true energy” is not directly available.

## What is in the dataset?

`data/nue_energy_demo.h5` contains 7,373 simulated electron-neutrino events between 0.5 and 5 GeV.

| Field | Shape | Meaning |
|---|---:|---|
| `images` | `(7373, 2, 100, 80)` | Two sparse detector views per event |
| `true_energy` | `(7373,)` | Simulated neutrino energy in GeV |
| `reco_energy` | `(7373,)` | Reference reconstruction included for comparison |
| `interaction` | `(7373,)` | Interaction category: QE, RES, DIS, or Other |
| `split` | `(7373,)` | `0=train`, `1=validation`, `2=test` |
| `event_id/*` | `(7373,)` | Event identifiers used to keep records aligned |

The fixed split contains 5,898 training events, 737 validation events, and 738 test events.

## What the notebook does

Open [`energy_reco.ipynb`](energy_reco.ipynb) and run it from top to bottom. It will:

1. inspect and validate the HDF5 data;
2. display the two detector views;
3. build a two-branch CNN, one branch for each view;
4. train with mean absolute error (MAE);
5. save the trained model as `models/energy_cnn.keras`;
6. compare the CNN with a constant baseline and the reference reconstruction;
7. examine global error, energy-dependent bias, resolution, and individual events.

## Example result from the included split

With the fixed seed and TensorFlow 2.15.1, the tested notebook gives approximately:

| Test metric | Reference reconstruction | CNN |
|---|---:|---:|
| MAE | 0.226 GeV | 0.190 GeV |
| RMSE | 0.355 GeV | 0.290 GeV |
| Central 68% fractional half-width | 0.117 | 0.103 |

The CNN reduces the average error and the broad tails, but it is not uniformly better at every energy. Most events lie near 2 GeV, so the network tends to pull rare low- and high-energy events toward that dense central region. This is a useful example of **regression toward the mean**.

## Run the tutorial

Python 3.11 is recommended.

```bash
conda create -n energy-reco python=3.11
conda activate energy-reco
pip install -r requirements.txt
jupyter lab energy_reco.ipynb
```

Training takes a few minutes on a modern laptop CPU. A GPU is optional.

## Questions to explore

- Why do the detector images contain so many zero-valued pixels?
- Why can a model have a better overall MAE but a larger bias in one energy range?
- What changes if the two detector views are exchanged or one view is removed?
- How would energy-balanced sampling affect rare low- and high-energy events?
- Why must a model trained on simulation be checked carefully before being used on real data?

## Limitations

- The dataset contains simulated electron-neutrino events only.
- The energy range is restricted to 0.5–5 GeV.
- The tutorial uses a compact teaching sample, not a full analysis dataset.
- The reference reconstruction is provided only as a comparison field.
- Detector and interaction-model systematic uncertainties are outside this introductory notebook.
- Performance on simulation does not establish performance on real detector data.

## Further reading

- J. Liu *et al.*, “Deep-Learning-Based Kinematic Reconstruction for DUNE,” [arXiv:2012.06181](https://arxiv.org/abs/2012.06181).
