# PathSnap | four-person hackathon workspace

AI-assisted reference implementation and actual development run. The organisers require independent team experimentation and explanations. Reproduce the results, investigate a design change of your own, document what each person did and disclose AI assistance as required. The PDFs give no prescribed four-person split or exact pre-lunch checklist. Each role folder includes its own handoff.

## Before lunch

1. Combine the four member folders in this directory (already done in the full project download). Open this folder in VS Code.
2. Extract the supplied Student folder.zip and locate the folder that directly contains train and student_test. Keep the original dataset outside the public repository.
3. Member 1 shows the inventory, a trip map and how ordered route labels are inferred. Member 2 shows the five-model CSV and trial history. Member 3 shows raw versus decoded results on a trip that improved and one that worsened. Member 4 starts the development demo. This is a suggested checkpoint.
4. Only the full project contains all dependencies between roles. Role ZIPs are handoffs within one project.

## Windows PowerShell setup on the model laptop

Use Python 3.12. From the extracted PathSnap project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PATHSNAP_DATA = 'E:\Hackathon\Student folder'
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
```

Replace the example data path with yours. Set these variables again in a new terminal. To reproduce the experiments, run in order:

```powershell
.\.venv\Scripts\python.exe -m member1_data.prepare
.\.venv\Scripts\python.exe -m member2_models.train
.\.venv\Scripts\python.exe -m member3_decoder.evaluate
.\.venv\Scripts\python.exe -m member4_demo.report
```

## Two-laptop demonstration

Laptop 1 on the same Wi-Fi as Laptop 2, using the setup above:

```powershell
.\.venv\Scripts\python.exe -m uvicorn member4_demo.api:app --host 0.0.0.0 --port 8000
```

The saved model lets you start the demo before retraining. Use ipconfig in a second terminal to find the Wi-Fi IPv4 address. Permit Python on your private network if Windows prompts. Laptop 2 only needs Python and member4_demo/web. From the project root:

```powershell
py -m http.server 8501 --bind 127.0.0.1 --directory member4_demo/web
```

On Laptop 2 open http://127.0.0.1:8501. Enter http://LAPTOP1-IP:8000, click Load trips, select a trip, then Reconstruct route. The map and its controls use the bundled Leaflet files. The API log on Laptop 1 shows the received endpoint and the result. Capture the real UI and receiving command window for the team report.

During final evaluation, follow the organiser's instructions for installing the reserved trips on Laptop 1. PATHSNAP_DATA points to the parent directory and PATHSNAP_SPLIT to the reserved partition name. Restart the API. Ensure the organiser's scoring convention agrees with the route-to-fix alignment or adapt the isolated scoring function. A missing reference gives an unavailable accuracy. The prediction model and UI receive no evaluation labels. Development reference layers are available only in the separate offline Folium map.

## Submission

Publish the team's reproducible work to one public GitHub repository after team review. Include source, requirements, weights, search configurations, evidence CSVs, figures and map-generation code. Submit its URL and one consolidated PDF. Replace the local demo evidence with screenshots from the actual two laptops. Use results/models.csv for the full comparison, results/bayesian_trials.csv for the search and results/decoder.csv for per-trip outcomes. The report generated from these files is a reference draft. Retain genuine team contributions and rerun after any changes.
