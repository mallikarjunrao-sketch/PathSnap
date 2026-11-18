# PathSnap

Matching noisy GPS traces to directed roads.

A single GPS reading often sits between two roads, especially at junctions and between parallel streets. If you just pick the closest road for every reading, the route jumps from one street to the next. PathSnap avoids this in two steps: a classifier scores the nearby roads for each reading, and a Viterbi decoder then uses how those roads connect to pick the most likely route for the whole trip.

On our ten student-test trips, the decoding step raised point accuracy from 81.81% to 85.80% and mean route continuity from 90.27% to 97.58%.

[Technical report](Technical_Report.pdf) | [Demo screenshots](screenshots/)

## Contents

- [The demo](#the-demo)
- [How it works](#how-it-works)
- [Results](#results)
- [What you need](#what-you-need)
- [Setup](#setup)
- [Using the demo](#using-the-demo)
- [API](#api)
- [Reproducing the experiments](#reproducing-the-experiments)
- [Project layout](#project-layout)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)

## The demo

The demo runs on two laptops connected to the same network.

- **Laptop 1** is the model server. It holds the dataset and the reference labels, loads the trained classifier and serves predictions through FastAPI.
- **Laptop 2** is the evaluator screen. In a browser, you pick a trip and see the GPS trace, the reconstructed route, point accuracy and route continuity on a map.

The reference labels never leave Laptop 1. Laptop 2 only receives the predicted geometry, the GPS trace and the scores.

## How it works

1. **Read the trip.** A trip has GPS observations, road nodes and directed road segments (arcs). A separate reference file lists the arcs the vehicle travelled, in order. A vehicle can spend dozens of readings on one arc, so we cannot pair the two files row by row. Instead, the training pipeline aligns the readings to the reference route.
2. **Score nearby roads.** Each reading gets six candidate arcs. The classifier scores each one using its distance from the reading, the direction of travel, the position along the arc and the time since the previous reading.
3. **Decode the sequence.** The Viterbi decoder combines those scores with road connectivity. Staying on an arc, or moving to a connected one, is treated as more plausible than jumping between roads that are not connected. A time gap over 15 seconds starts a new sequence.
4. **Return the result.** Laptop 1 uses the reference route to work out accuracy after decoding, then sends the prediction to Laptop 2.

## Results

We trained on up to 100 evenly spaced observations from each of 80 training trips. Three-fold cross-validation kept every trip inside a single fold. The student-test comparison used all 19,160 observations from ten separate trips.

| Classifier | Mean CV macro F1 | Student-test top-1 accuracy |
|---|---:|---:|
| Logistic regression | 0.8821 | 81.27% |
| Gaussian Naive Bayes | 0.8781 | 78.59% |
| Decision tree | 0.8860 | 81.60% |
| Extra Trees | 0.8876 | 81.98% |
| Histogram gradient boosting | 0.8898 | 81.81% |

Extra Trees had the best student-test top-1 accuracy. We chose histogram gradient boosting because it had the highest mean cross-validation macro F1, which was our selection criterion. Seven Bayesian optimisation trials tested the maximum number of leaves per tree. The best value was 15, the same as the baseline, so tuning did not improve accuracy.

| Student-test trips | Point accuracy | Mean continuity |
|---|---:|---:|
| Classifier only | 81.81% | 90.27% |
| Classifier and Viterbi | 85.80% | 97.58% |

The gain is not the same on every trip. On trip `00000007`, continuity improved but point accuracy dropped from 91.73% to 88.26%. A route can follow connected roads and still be on the wrong street.

In the final evaluator run, the saved model and decoder were used on ten reserved trips. Across them, 15,909 of 19,674 GPS observations matched the aligned reference road, which is 80.86% overall point accuracy under the project's scoring method.

More detail: [model comparison](results/models.csv), [Bayesian search trials](results/bayesian_trials.csv), [per-trip decoder results](results/decoder.csv)

## What you need

- Windows on both laptops.
- Laptop 1: Python 3.12 installed with the Windows Python launcher. The first run needs internet access to install dependencies.
- Laptop 2: Python 3.8 or newer. Internet access is optional and only needed for OpenStreetMap street labels. The ten student-test trips also come with local road outlines that work offline.
- Both laptops on the same Wi-Fi or hotspot.
- The dataset from the organisers, which is supplied separately and is not included in this repository.

## Setup

### Laptop 1: model server

1. In the repository root, create a folder called `data`.
2. Extract the organisers' dataset and put the `train` and `student_test` folders directly inside `data`. For example, this file should exist:

   ```
   data/student_test/00000000/00000000.track
   ```

3. Double-click `START_SERVER.cmd`. The first run installs dependencies. The launcher loads the saved classifier from `artifacts/model.joblib`.
4. Wait for **Application startup complete** and leave the terminal open.
5. The terminal prints one or more addresses. Copy the one for your active Wi-Fi or hotspot adapter, including the port (`8000`). If you are not sure which one it is, run `ipconfig` and look at the active adapter.

### Laptop 2: evaluator screen

1. Connect to the same Wi-Fi or hotspot as Laptop 1.
2. Copy the whole `Laptop_2_UI` folder to Laptop 2, including its `web` subfolder.
3. Double-click `START_UI.cmd`. Your browser opens at `http://127.0.0.1:8501`.

## Using the demo

On Laptop 2:

1. Type the address from Laptop 1 into the **Model laptop API** field, for example `http://172.20.10.4:8000`.
2. Click **Load trips**.
3. Pick a trip from the **Evaluation trip** list.
4. Click **Reconstruct route**.

The screen shows the number of GPS fixes, the point accuracy and the route continuity, along with a map. Blue dashes trace the GPS readings in time order, green lines are the predicted road segments, and gray lines are local roads for the supplied demo trips. Street names and landmarks come from OpenStreetMap when you are online.

### Running the reserved evaluator trips

1. Put the supplied `evaluator_test` folder inside `data` on Laptop 1.
2. Stop the server if it is running.
3. In PowerShell, from the repository root, run:

   ```powershell
   $env:PATHSNAP_SPLIT = 'evaluator_test'
   .\START_SERVER.cmd
   ```

4. On Laptop 2, click **Load trips** again.

### Exploring the development map

Open `results/development_map.html` in a browser. It is an interactive Folium map of trip `00000007` and works offline. The layer controls let you switch the GPS observations, road network, reference route and prediction on and off. Two arcs running in opposite directions can share exactly the same geometry, so turn the prediction off to see the reference underneath it.

## API

Laptop 1 serves a FastAPI app on port `8000`.

| Endpoint | What it does |
|---|---|
| `GET /trips` | Lists the trips in the current split. |
| `GET /predict/{trip_id}` | Builds candidates, scores them, decodes the route and returns the predicted geometry, GPS coordinates and aggregate scores. Example: `GET /predict/00000000`. |

## Reproducing the experiments

From the repository root on Laptop 1, run `RUN_ON_WINDOWS.cmd`. It prepares the data, trains all five classifiers, repeats the Bayesian search and opens the comparison page.

Once training has finished, you can rebuild the per-trip metrics, the decoder comparison, the Folium map and the report with:

```powershell
$env:OMP_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m member2_models.observation_f1
.\.venv\Scripts\python.exe -m member3_decoder.evaluate
.\.venv\Scripts\python.exe docs/build_report.py
```

## Project layout

| Path | Contents |
|---|---|
| `PES1UG24AM153_data/` | Trip loading, data checks, reference alignment and candidate features |
| `PES1UG24AM155_models/` | The five classifiers, grouped cross-validation, Bayesian optimisation and evaluation metrics |
| `PES1UG24AM156_decoder/` | Viterbi decoding, sequence evaluation and Folium map generation |
| `PES1UG24AM154_demo/` | FastAPI endpoints for listing trips and reconstructing a selected route |
| `Laptop_2_UI/` | Browser interface, Windows launcher, Leaflet assets and offline road layers |
| `artifacts/` | The saved classifier and the selected hyperparameter configuration |
| `results/` | Experiment tables, confusion matrices, logs, figures and the interactive map |
| `docs/` | Technical report, report-generation script and demo screenshots |

## Troubleshooting

**Load trips fails on Laptop 2.** Check that both laptops are on the same Wi-Fi or hotspot and that the address starts with `http://` and ends with `:8000`. Use the address printed for the active adapter, not `0.0.0.0`. If Windows asked about network access when the server started, it needs to be allowed on private networks.

**The trip list is empty.** Make sure `train` and `student_test` sit directly inside `data`. If you set `PATHSNAP_SPLIT`, the folder it names must exist there too.

**The first launch is slow or fails.** The first run installs dependencies, so it needs internet access.

**The map has no street names.** Labels come from OpenStreetMap and need internet access. The student-test trips still show local road outlines offline.

**The prediction and reference lines look identical on the map.** Opposite-direction arcs can share geometry. Use the accuracy and continuity scores alongside the map.

## Known limitations

- Point accuracy depends on how well the readings can be aligned to the reference route. Parallel roads, stops and repeated visits can make that alignment ambiguous.
- In 0.79% of student-test fixes, the reference arc is not among the six candidates, so no ranking can recover it.
- A smoother route is not always a more accurate one, as trip `00000007` shows.
- Hyperparameter tuning gave no improvement over the 15-leaf baseline.
- The setup instructions cover Windows only.

## Data

The raw trip data is supplied separately by the organisers. The screenshots in `docs/screenshots/` come from our two-laptop demonstration.

## Project Structure

| Path | Description |
|---|---|
| `PES1UG24AM153_data/` | Data loading, EDA, cleaning and feature engineering |
| `PES1UG24AM155_models/` | Model training, comparison and Bayesian hyperparameter optimisation |
| `PES1UG24AM156_decoder/` | HMM and Viterbi route reconstruction |
| `PES1UG24AM154_demo/` | Demo / integration of the end-to-end pipeline |
| `artifacts/` | Saved model weights and preprocessing artifacts |
| `results/` | Experiment results, metrics CSVs and plots |
| `screenshots/` | UI and API screenshots used in the report |
| `web/` | Evaluator-facing UI |
| `launch.py` | Entry point to launch the system |
| `show_results.py` | Prints / displays saved evaluation results |
| `START_SERVER.cmd` / `START_UI.cmd` | Windows scripts to start the API (Laptop 1) and UI (Laptop 2) |
| `RUN_ON_WINDOWS.cmd` | One-click run script for Windows |
| `requirements.txt` | Python dependencies |
| `Technical_Report.pdf` | Final technical report |
