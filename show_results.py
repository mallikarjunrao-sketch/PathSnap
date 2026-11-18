from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from member1_data.trips import ROOT, load_trip

out = Path("results")
inventory = pd.read_csv(out/"inventory.csv")
models = pd.read_csv(out/"models.csv")
trip = load_trip(ROOT/"train"/"00000002")
route = np.loadtxt(trip.folder/"00000002.route", dtype=int)
fig, axes = plt.subplots(1, 2, figsize=(11,4), layout="constrained")
ax=axes[0]
for ids,color,width,label in [(np.arange(len(trip.arcs)),"#b8beb8",.5,"Network"),(route,"#df8614",3,"Reference")]:
    ax.add_collection(LineCollection(trip.net[trip.arcs[ids]],colors=color,linewidths=width,label=label))
ax.scatter(*trip.xy.T,s=2,color="#275b88",label="GPS",zorder=3)
ax.set(xlim=(trip.xy[:,0].min()-80,trip.xy[:,0].max()+80),ylim=(trip.xy[:,1].min()-80,trip.xy[:,1].max()+80),aspect="equal",xlabel="East (m)",ylabel="North (m)",title="Training trip 00000002")
ax.legend(fontsize=8)
ax=axes[1]
for split,g in inventory.groupby("split"):
    ax.scatter(np.sort(g.fixes),np.arange(1,len(g)+1)/len(g),label=split,s=15)
ax.set(xscale="log",xlabel="GPS observations per trip (log scale)",ylabel="Fraction of trips at or below this size",title="Trip sizes across both partitions")
ax.legend()
fig.savefig(out/"eda.png",dpi=150)
base=models[~models.model.str.endswith("_tuned")]
summary=inventory.groupby("split").agg(trips=("trip","count"),GPS_fixes=("fixes","sum"),missing_GPS_cells=("missing","sum"),largest_gap_seconds=("max_gap_s","max"))
winner=json.loads(Path("artifacts/selection.json").read_text())["winner"]
selected=base[base.model==winner].iloc[0]
spec = json.loads(Path("artifacts/selection.json").read_text())
trials = pd.read_csv(out/"bayesian_trials.csv")
metric_cols = ["cv_macro_f1","macro_f1","top1","log_loss","inference_s"]
labels = {"logistic":"Logistic regression","gaussian_nb":"Gaussian Naive Bayes","tree":"Decision tree","extra_trees":"Extra Trees","hist_boost":"Histogram gradient boosting"}
base = base.copy()
base["model"] = base["model"].map(labels)
before_after = models[models.model.isin([winner,winner+"_tuned"])][metric_cols].copy()
before_after.index = ["Before tuning","After tuning"]
fig,ax = plt.subplots(figsize=(10,4),layout="constrained")
base.set_index("model")[["cv_macro_f1","macro_f1","top1"]].rename(columns={"cv_macro_f1":"Grouped CV macro F1","macro_f1":"Student macro F1","top1":"Student top-1"}).plot.barh(ax=ax)
ax.set(xlim=(0,1),ylabel="",xlabel="Score")
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.15),ncol=3,fontsize=9)
fig.savefig(out/"model_comparison.png",dpi=150)
page=f"""<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PathSnap checkpoint</title>
<style>body{{max-width:960px;margin:35px auto;padding:0 18px;color:#203630;font:17px/1.55 system-ui}}h2{{margin-top:30px}}table{{border-collapse:collapse}}td,th{{padding:6px 12px;text-align:left;border-bottom:1px solid #ccd5cc}}img{{width:100%}}</style>
<h1>PathSnap • EDA and model training</h1><h2>1. Explore the trips</h2>{summary.to_html(border=0)}
<img src="results/eda.png" alt="Training GPS map and cumulative trip sizes"><p>Each trip has a GPS track, directed road segments, road nodes and an ordered reference route. A reference row describes a traversed segment. Multiple GPS observations can correspond to it. The alignment estimates point-level labels from the ordered route.</p>
<h2>2. Compare five classifiers</h2><p>Training used up to 100 evenly spaced fixes per trip from all 80 training trips. For each fix the models scored six candidate roads. Three-fold cross-validation kept whole trips together. Student-test evaluation used every fix on the ten supplied student-test trips.</p>
{base[['model','cv_macro_f1','macro_f1','top1','log_loss']].rename(columns={'cv_macro_f1':'CV macro F1','macro_f1':'Student macro F1','top1':'Student top-1','log_loss':'Log loss (candidate present)'}).to_html(index=False,border=0,float_format=lambda x:f'{x:.4f}')}
<img src="results/model_comparison.png" alt="Grouped CV and student-test scores for the five models"><p><strong>{labels[winner]}</strong> had the highest mean grouped CV macro F1 ({selected.cv_macro_f1:.4f}). It selected the aligned reference road on {selected.top1:.2%} of student-test GPS observations. These scores describe the classifier before route decoding.</p>
<h2>3. Bayesian hyperparameter tuning</h2><p>Selected model: {labels[winner]}. Parameter: <code>{spec['parameter']}</code>. Search space: {spec['domain']}. The first three trials sampled the range. A Gaussian process then used the previous scores to select four further settings by expected improvement. Each setting used the same three training folds.</p>
{trials[['trial','value','cv_macro_f1']].rename(columns={'value':spec['parameter'],'cv_macro_f1':'Grouped CV macro F1'}).to_html(index=False,border=0,float_format=lambda x:f'{x:.5f}')}
<h3>Before and after tuning</h3>{before_after.rename(columns={'cv_macro_f1':'CV macro F1','macro_f1':'Student macro F1','top1':'Student top-1','log_loss':'Log loss','inference_s':'Inference (s)'}).to_html(border=0,float_format=lambda x:f'{x:.4f}')}
<p>Best tested setting: <strong>{spec['parameter']} = {spec['value']}</strong>. Change in student top-1 accuracy: {(before_after.iloc[1].top1-before_after.iloc[0].top1)*100:+.2f} percentage points.</p>
<p><a href="results/inventory.csv">Trip inventory</a> · <a href="results/models.csv">Experiment results</a> · <a href="results/bayesian_trials.csv">Tuning trials</a></p>
<p>AI-assisted reference run on the supplied development data.</p>"""
Path("OPEN_CHECKPOINT.html").write_text(page,encoding="utf-8")
