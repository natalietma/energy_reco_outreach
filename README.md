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

You do not need a GPU for this tutorial. A normal laptop is enough.

Enter the commands below **one line at a time** in Terminal on macOS/Linux or Anaconda Prompt on Windows.

### Step 1: Download the repository

Using Git:

```bash
git clone https://github.com/natalietma/neutrino-energy-reconstruction-tutorial.git
cd neutrino-energy-reconstruction-tutorial
```

If you do not have Git, click **Code → Download ZIP** on this GitHub page, unzip the downloaded file, and open Terminal inside the extracted folder.

### Step 2: Check that you are in the correct folder

Run:

```bash
pwd
ls
```

You should see files and folders similar to:

```text
README.md
energy_reco.ipynb
requirements.txt
data/
figures/
models/
```

If you do not see `energy_reco.ipynb`, use `cd` to enter the correct project folder before continuing.

### Step 3: Create a Python environment

Python 3.11 is recommended.

If you have Anaconda or Miniconda, run:

```bash
conda create -n energy-reco python=3.11 -y
conda activate energy-reco
```

Your terminal should now begin with something similar to:

```text
(energy-reco)
```

This means that the new environment is active.

### Step 4: Install the required packages

Run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

This installs TensorFlow, NumPy, Matplotlib, HDF5 support, and JupyterLab.

The installation may take several minutes.

### Step 5: Check the dataset

Run:

```bash
python -c "import h5py; f=h5py.File('data/nue_energy_demo.h5','r'); print('Images:',f['images'].shape); print('True energy:',f['true_energy'].shape); f.close()"
```

The output should include:

```text
Images: (7373, 2, 100, 80)
True energy: (7373,)
```

Each event contains two detector images with 100 × 80 pixels.

### Step 6: Open the Jupyter notebook

Run:

```bash
jupyter lab energy_reco.ipynb
```

JupyterLab should open automatically in your web browser.

If the browser does not open, copy the `http://localhost:...` URL displayed in Terminal and paste it into your browser.

> Keep the Terminal window open while using JupyterLab.

### Step 7: Run the tutorial

In JupyterLab:

1. Open `energy_reco.ipynb`.
2. Select **Run → Run All Cells** from the top menu.
3. If asked to choose a kernel, select **Python 3**.
4. Wait while the CNN trains and evaluates the test events.

A number such as `[5]` beside a cell means that the cell has finished running. An asterisk `[*]` means that the cell is still running.

Training usually takes several minutes on a laptop CPU. A GPU is optional.

### Step 8: Find the results

After the notebook finishes:

- the trained model is saved to `models/energy_cnn.keras`;
- reconstruction figures are saved in `figures/`;
- the final notebook cells print the CNN and traditional-reconstruction metrics.

The main output figures include:

```text
figures/validation_loss_over_epochs.png
figures/nue_energy_fractional_error.png
figures/nue_energy_prediction_bias.png
figures/nue_energy_resolution.png
figures/predicted_vs_true.png
figures/event_examples.png
```

### Step 9: Stop JupyterLab

Return to Terminal and press:

```text
Control + C
```

Confirm shutdown if Terminal asks for permission.

### Open the tutorial again later

You do not need to reinstall the packages. Open Terminal and run:

```bash
cd neutrino-energy-reconstruction-tutorial
conda activate energy-reco
jupyter lab energy_reco.ipynb
```

### Common problems

#### `conda: command not found`

Install Anaconda or Miniconda, close Terminal, and open it again.

#### `jupyter: command not found`

Activate the environment and reinstall the requirements:

```bash
conda activate energy-reco
python -m pip install -r requirements.txt
```

#### `FileNotFoundError: data/nue_energy_demo.h5`

You are probably running Jupyter from the wrong folder. Stop Jupyter, enter the repository folder, and start it again:

```bash
cd neutrino-energy-reconstruction-tutorial
jupyter lab energy_reco.ipynb
```

#### A notebook cell shows an error

Select **Kernel → Restart Kernel and Run All Cells**. Run the notebook from the beginning because later cells depend on variables created by earlier cells.

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
