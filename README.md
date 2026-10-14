# Operando Optical Machine Vision: DOOM, BOOMHER and Machine-Learning Tutorials

This repository contains the code developed as part of my PhD research into the use of machine learning (ML) and machine vision (MV) to extract quantitative information from operando optical microscopy of aqueous zinc batteries.

The repository contains three main components:

1. **DOOM (Dendrite Optical Observation Method)** — software for analysing visible zinc-electrode surface morphology using image segmentation.
2. **BOOMHER (Bubble Optical Observation Methods for the Hydrogen Evolution Reaction)** — software for detecting, measuring and tracking visible gas bubbles in time-lapse optical images.
3. **Machine-learning tutorials** — worked examples covering image and tabular classification, regression, image segmentation and object detection.


---

## Important: trained models must be downloaded separately

The trained ML models used by DOOM and BOOMHER are provided separately through the **Releases** section of this GitHub repository.

Three model files are required:

| Software | Model file | Required location |
|---|---|---|
| DOOM | `DOOM_Surface_model.pkl` | `DOOM_code/models/DOOM_Surface_model.pkl` |
| DOOM | `DOOM_bubble_model.pt` | `DOOM_code/models/DOOM_bubble_model.pt` |
| BOOMHER | `BOOMHER_model.pt` | `BOOMHER_code/BOOMHER_model.pt` |

After cloning or downloading the repository, download the model files from the [GitHub Releases](../../releases) section.

The filenames and locations should be retained exactly unless the corresponding paths in the code are also changed.

---

# DOOM

## Dendrite Optical Observation Method

DOOM is a Jupyter-notebook-based workflow for analysing visible changes in zinc-electrode surface morphology from operando optical time-lapse datasets.

The workflow uses two ML models.

The first is an image-segmentation model that classifies the visible electrode surface pixel-by-pixel into:

- **Dendrites**
- **Not Dendrites**

The resulting segmentation masks are used to calculate the fraction of the visible electrode surface assigned to each class through time.

The second model is a YOLO object-detection model used to identify visible bubbles. Pixels contained within detected bubble regions are excluded from the surface analysis so that bubbles are not incorrectly treated as electrode morphology.

DOOM therefore converts an optical image sequence into time-resolved measurements of visible electrode-surface evolution that can be compared with the corresponding electrochemical cycling data.

### Current DOOM version

The version included in this repository is the version prepared for use by other researchers at the industry partner.

It is not the exact working version that I personally used to prepare the analyses presented in my thesis. My own analysis code changed repeatedly as the project developed and contains additional experimental and analysis-specific modifications.

The version provided here was instead prepared as a more self-contained workflow that could be used by another researcher without requiring the collection of working notebooks and temporary code used during development.

It should therefore be treated as research software rather than a polished general-purpose application.

### DOOM model setup

Download the following files from the [GitHub Releases](../../releases) section:

```text
DOOM_Surface_model.pkl
DOOM_bubble_model.pt
```

Place both files inside:

```text
DOOM/models/
```

The resulting structure should be:

```text
DOOM/
├── controlpipeline1.ipynb
├── fullpipeslave1.ipynb
├── models/
│   ├── DOOM_Surface_model.pkl
│   └── DOOM_bubble_model.pt
└── mainpipeline/
```

The current DOOM code expects the models at:

```text
DOOM/models/DOOM_Surface_model.pkl
DOOM/models/DOOM_bubble_model.pt
```

### Running DOOM

The main notebook is:

```text
DOOM/controlpipeline1.ipynb
```

This notebook controls the analysis, including selection of the image dataset, starting and ending images, and image-sampling frequency.

The individual image-processing operations are performed using:

```text
DOOM/fullpipeslave1.ipynb
```

The processing workflow creates the required intermediate folders automatically under:

```text
DOOM/mainpipeline/
```

These folders are used to store temporary image crops, segmentation outputs and reconstructed masks.

### DOOM outputs

DOOM produces both numerical and visual outputs.

For each processed image, the segmentation model produces a full-image classification mask showing the regions classified as dendritic and non-dendritic surface. Bubble regions identified by the object-detection model are excluded from the surface calculation.

Across the full image sequence, these masks are converted into time-resolved measurements of fractional dendrite coverage.

The main outputs therefore include:

- full-image segmentation masks;
- fractional dendrite and non-dendrite surface coverage;
- bubble-excluded surface measurements;
- time-aligned electrochemical data; and
- dashboard visualisations combining the original images, segmentation results and electrochemical measurements.

The dashboard output provides a convenient way to review both the extracted data and the quality of the ML classifications throughout the experiment.

An example animated DOOM dashboard can be viewed here:

[DOOM animated dashboard – YouTube](https://www.youtube.com/watch?v=tDI2ZKNRhEI)

### DOOM experimental data

The input experimental data should be organised with an `images` folder and the corresponding cycling-data file:

```text
experiment_folder/
├── images/
│   ├── image_YY-MM-DD-HH-MM-SS_....png
│   ├── image_YY-MM-DD-HH-MM-SS_....png
│   └── ...
└── cyclingdata.xlsx
```

The timestamps contained in the image filenames are used to align the image-derived measurements with the electrochemical cycling data.

---

# BOOMHER

## Bubble Optical Observation Methods for the Hydrogen Evolution Reaction

BOOMHER is a Python-script-based workflow for detecting, measuring and tracking visible gas bubbles across operando optical time-lapse datasets.

Unlike DOOM and the machine-learning tutorials, the distributed BOOMHER workflow is a Python script rather than a Jupyter notebook.

The workflow combines:

- YOLO object detection;
- machine-vision measurements; and
- object tracking between sequential images.

The YOLO model identifies visible bubbles and produces a bounding box around each detection. These bounding boxes are then used to estimate bubble position and visible dimensions.

An object-tracking system associates detections between sequential images and assigns an ID to individual bubbles. This allows the same visible bubble to be followed through time rather than treating every detection in every image as an independent object.

BOOMHER can extract measurements including:

- number of visible bubbles;
- bubble position;
- visible bubble size;
- estimated bubble volume;
- surface coverage;
- growth of individual bubbles;
- cumulative visible bubble growth; and
- changes in bubble behaviour through time.

### Current BOOMHER version

The version included in this repository is the standalone Python version deployed at the industry partner.

It is far from a perfect or finished piece of software, but it provides a relatively self-contained implementation of the BOOMHER workflow that can be run by another user.

This is not the version I personally use for most ongoing analysis. My working implementation is spread across a heavily modified set of Jupyter notebooks that I have continued to change as new experiments, analyses and figures have been developed.

That working version is more flexible for my own research but considerably less suitable for distribution. The Python version provided here therefore represents the more stable user-facing implementation rather than the continuously modified research version.

### BOOMHER model setup

Download:

```text
BOOMHER_model.pt
```

from the [GitHub Releases](../../releases) section.

Place the model directly inside the main BOOMHER folder:

```text
BOOMHER/
├── Main_BOOMHER_script.py
├── BOOMHER_model.pt
└── functionstorage/
    ├── Create_bubble_ID.py
    ├── MachineVisionFunction.py
    ├── ProccessFunction3.py
    ├── SetupDataframeFunction.py
    └── WritetoDataframesFunction.py
```

The current BOOMHER processing code loads:

```python
YOLO("BOOMHER_model.pt")
```

so the model must remain in the BOOMHER working directory unless this path is changed manually.

The `functionstorage` directory must also remain in its current location because `Main_BOOMHER_script.py` imports the supporting analysis functions from this folder.

### BOOMHER outputs

BOOMHER produces both numerical measurements and visual outputs from the analysed image sequence.

The numerical outputs record properties of the detected bubbles through time, including their size, position, assigned ID and estimated volume.

The program also generates derived measurements and visualisations including:

- bubble surface coverage;
- bubble-volume measurements;
- individual bubble tracking data;
- bubble-growth measurements;
- bubble-position data;
- bubble-residence heatmaps;
- electrochemical cycling plots;
- annotated images showing detected bubbles and their assigned IDs; and
- an animated dashboard combining the image and quantitative outputs.

Several of the intermediate and processed datasets are also saved as CSV files so they can be examined independently or used for additional analysis in other software.

The animated dashboard is intended both as a presentation of the extracted measurements and as a way of reviewing whether the model detections and tracking behaviour remain sensible throughout the experiment.

An example animated BOOMHER dashboard can be viewed here:

[BOOMHER animated dashboard – YouTube](ADD_YOUTUBE_LINK_HERE)

### BOOMHER experimental data

The current BOOMHER workflow expects an experimental data folder containing:

```text
experiment_folder/
├── images/
│   ├── image_YY-MM-DD-HH-MM-SS_....png
│   ├── image_YY-MM-DD-HH-MM-SS_....png
│   └── ...
└── cyclingdata.csv
```

The image sequence is analysed by BOOMHER and the timestamps contained in the filenames are used to associate the resulting measurements with experimental time and cycling data.

---

# Machine-learning tutorials

The machine-learning tutorials are included primarily so that readers of my thesis can reproduce the basic process used to train, evaluate and apply the types of ML models used throughout this project.

DOOM and BOOMHER demonstrate how ML models can be incorporated into larger analysis pipelines, but they are not ideal examples for learning the ML components themselves. In both programs, the actual ML operations are surrounded by additional systems for image processing, file handling, object tracking, data organisation and visualisation.

Pulling those programs apart simply to understand how the underlying ML works would therefore be unnecessarily difficult.

The tutorial material instead isolates the ML workflows from those larger systems.

Each tutorial is provided as a **Jupyter notebook** with extensive written explanations alongside the code describing what the different parts do, why they are required, and how they fit together.

The examples are intentionally general rather than being specific to batteries or operando microscopy. Synthetic datasets are provided so that each notebook can be run immediately without requiring access to any experimental datasets from the thesis.

Because the examples are deliberately simple and self-contained, they are intended to be easy to reverse engineer and modify. A user can replace the supplied synthetic inputs and labels with their own data and use the notebooks as starting templates for different applications.

The aim is therefore not simply to provide code that reproduces one specific model, but to make the underlying workflows sufficiently accessible that another researcher could use them to train, evaluate and apply their own ML models.

---

## Image-based tutorials

The image tutorials cover four common ML tasks.

### Classification

Training a model to assign an entire image to a class.

### Regression

Training a model to predict a continuous numerical value from an image.

### Image segmentation

Training a model to assign a class to individual pixels and generate a segmentation mask.

This is the general type of ML method used for electrode-surface classification in DOOM.

### Object detection

Training a model to locate and classify objects using bounding boxes.

The tutorial uses Ultralytics YOLO and demonstrates the same general type of ML method used for bubble detection in BOOMHER.

---

## Tabular-data tutorials

The tabular tutorials cover:

- **Classification**
- **Regression**

The classification tutorial also demonstrates how to use a custom neural-network model rather than relying entirely on FastAI's standard tabular model.

This example shows how values can be extracted from a Pandas DataFrame, converted into PyTorch tensors and supplied to a user-defined model architecture.

The purpose is to demonstrate how the normal workflow can be extended when greater control over the input data or model architecture is required.

---

# General tutorial workflow

Although the exact implementation differs between the different ML tasks, the tutorials follow the same general structure:

1. **Import and organise the data**
2. **Set up the model and data pipeline**
3. **Train the model**
4. **Evaluate how well the model has learned the task**
5. **Apply the trained model to new data**

Keeping the same broad structure across the tutorials makes it easier to identify which parts need to be changed when adapting an example to another dataset or application.

---

# Synthetic example data

The tutorials include scripts that generate the example datasets required to run them.

These generators create actual files that are subsequently loaded by the tutorial notebooks, allowing the examples to reproduce the basic structure of a real ML workflow without requiring an external dataset.

## Image datasets

For the image-based tutorials, the generation scripts create synthetic images containing simple geometric shapes.

Depending on the task, the scripts also generate the corresponding labels required for training.

For example:

- **Classification** generates images together with the class assigned to each image.
- **Regression** generates images together with numerical properties of the generated shapes that can be used as regression targets.
- **Image segmentation** generates both the input images and the corresponding pixel-level mask images.
- **Object detection** generates the input images and the corresponding bounding-box label files required by YOLO.

The generated files therefore reproduce the basic structure of labelled image datasets used in real applications: images are stored as image files while their corresponding classes, numerical targets, masks or bounding boxes provide the correct outputs used during training.

## Tabular datasets

The tabular examples instead use representative CSV files containing synthetic numerical and categorical data.

These CSV files are read into the notebooks using Pandas and converted into DataFrames in the same general way that experimental tabular datasets would normally be loaded.

This makes the tabular tutorials directly comparable with real workflows in which experimental measurements are stored in spreadsheets or CSV files and then supplied to an ML model.

The relationships in the synthetic datasets are intentionally simple so that the structure of the data and the behaviour of the trained model are easy to understand.

The generated datasets can then be replaced with a user's own images, masks, labels or CSV files when adapting the tutorials to another application.

---

# Required Python libraries

The Python libraries required to run the code are listed in:

```text
requirements.txt
```

After cloning or downloading the repository, the required libraries can be installed from the repository root using:

```bash
pip install -r requirements.txt
```

This installs the Python dependencies used across the tutorials, DOOM and BOOMHER.

A CUDA-compatible NVIDIA GPU is recommended for model training and for processing large image datasets.

The smaller tutorial examples can also be run on a CPU, although model training and image processing will generally be slower.

---

# Quick setup

Clone or download this repository.

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

Then download the trained models from the [GitHub Releases](../../releases) section and ensure that the following files exist:

```text
DOOM/models/DOOM_Surface_model.pkl
DOOM/models/DOOM_bubble_model.pt
BOOMHER/BOOMHER_model.pt
```

For the tutorials, open the relevant Jupyter notebook and run the cells sequentially.

For DOOM, begin with:

```text
DOOM/controlpipeline1.ipynb
```

For BOOMHER, run:

```text
BOOMHER/Main_BOOMHER_script.py
```

---

# Using the supplied models on other datasets

The supplied DOOM and BOOMHER models were trained using the operando optical datasets collected during this project.

Their performance therefore depends on the visual conditions represented in those training datasets, including:

- image scale;
- illumination;
- optical configuration;
- electrode appearance;
- image contrast;
- bubble appearance; and
- electrode morphology.

The supplied models should not be assumed to generalise reliably to substantially different experimental or imaging conditions.

The DOOM and BOOMHER analysis pipelines can still be adapted to other datasets, but new representative training data and retrained ML models may be required.

The machine-learning tutorials included in this repository provide simpler starting points for users who want to train replacement models for their own data.

---

# Relationship to the PhD thesis

This repository contains supporting software and tutorial material developed as part of my PhD research into machine-learning-enabled analysis of operando optical microscopy of aqueous zinc batteries.

DOOM and BOOMHER were developed to convert large optical time-lapse datasets into structured quantitative measurements:

- **DOOM** uses image segmentation and related machine-vision analysis to quantify visible zinc-electrode surface evolution.
- **BOOMHER** uses object detection and object tracking to quantify visible bubble behaviour.

The accompanying tutorials provide simplified and more general examples of the machine-learning methods used to develop these tools, allowing the underlying model-training and deployment workflows to be examined independently of the larger DOOM and BOOMHER analysis systems.
