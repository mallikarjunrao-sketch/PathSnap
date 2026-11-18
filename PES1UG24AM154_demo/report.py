"""Rebuild the reference PDF and charts from the saved experiment records."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from member1_data.trips import ROOT,load_trip

def main():
    out=Path("results")
    m=pd.read_csv(out/"models.csv")
    trials=pd.read_csv(out/"bayesian_trials.csv")
    dec=pd.read_csv(out/"decoder.csv",dtype={"trip":str})
    inv=pd.read_csv(out/"inventory.csv",dtype={"trip":str})
    spec=json.loads(Path("artifacts/selection.json").read_text())
    names=["Logistic","Gaussian NB","Tree","Extra Trees","Hist. boost","Tuned"]
    fig,ax=plt.subplots(1,2,figsize=(10,3.6),layout="constrained")
    y=np.arange(len(names))
    for offset,col,label in [(-.22,"cv_macro_f1","CV F1"),(0,"macro_f1","Student F1"),(.22,"top1","Student top-1")]:
        ax[0].barh(y+offset,m[col],height=.2,label=label)
    ax[0].set(yticks=y,yticklabels=names,xlim=(0,1),xlabel="Score");ax[0].legend(fontsize=7,loc="lower left")
    ax[1].barh(names,m.log_loss,color="#36745b");ax[1].set_xlabel("Conditional observation log loss")
    fig.savefig(out/"comparison.png",dpi=155);plt.close(fig)
    t=load_trip(ROOT/"student_test"/"00000007")
    import joblib
    from member1_data.trips import features
    from member2_models.train import scores
    from member3_decoder.decode import decode
    ids,X=features(t,np.arange(len(t.gps)))
    path=decode(t.arcs,ids,scores(joblib.load("artifacts/model.joblib"),X),t.gps[:,2])
    ref=np.loadtxt(t.folder/(t.folder.name+".route"),int,ndmin=1)
    fig,ax=plt.subplots(figsize=(8.8,4.2),layout="constrained")
    for label,edges,color,width in [("Road network",t.arcs,"#c4cbc8",.55),("Reference",t.arcs[ref],"#e59a46",4),("Prediction",t.arcs[path],"#287457",1.7)]:
        ax.add_collection(LineCollection(t.net[edges],colors=color,linewidths=width,label=label))
    ax.scatter(*t.xy.T,s=1,color="#424e9e",label="GPS",zorder=5)
    for limit,col in [(ax.set_xlim,0),(ax.set_ylim,1)]: limit(t.xy[:,col].min()-120,t.xy[:,col].max()+120)
    ax.set_aspect("equal");ax.legend(fontsize=8);ax.set(xlabel="East (m)",ylabel="North (m)")
    fig.savefig(out/"geography.png",dpi=165);plt.close(fig)
    styles=getSampleStyleSheet();story=[]
    def h(s): story.append(Paragraph(s,styles["Heading1"]))
    def sub(s): story.append(Paragraph(s,styles["Heading3"]))
    def p(s):story.extend([Paragraph(s,styles["BodyText"]),Spacer(1,5)])
    def table(rows):
        t=Table(rows,hAlign="LEFT",repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#204c40")),("TEXTCOLOR",(0,0),(-1,0),colors.white),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),8),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#edf3ef")])]))
        story.extend([t,Spacer(1,8)])
    h("PathSnap | Development reference run")
    p("AI-assisted implementation and measured reference experiments. The team should reproduce the work, record its own decisions and add its live deployment evidence before submission.")
    sub("Dataset and preprocessing")
    table([["Partition","Trips","GPS fixes","Missing GPS cells","Max gap (s)"]]+[[k,len(g),f"{g.fixes.sum():,}",g.missing.sum(),int(g.max_gap_s.max())] for k,g in inv.groupby("split")])
    p("Trip-local .nodes rows define coordinates and .arcs rows define directed edges. GPS fixes and ordered reference arcs have different lengths. A dynamic program assigns fixes to nondecreasing route positions, minimising summed point-to-segment distances with a weak progress penalty. This gives approximate point labels. Ambiguous parallel roads and repeated visits can change the alignment.")
    p("Both partitions had strictly increasing timestamps and no isolated nodes. Weakly disconnected components occurred, with up to 421 in one training network. Training trip 00000090 contains a 153,064-second gap. The per-trip inventory includes extents, file sizes, counts and connectivity.")
    sub("Features and leakage controls")
    p("Scale longitude by the cosine of mean trip latitude and convert both axes to local metres. Index road segments with anchors spaced at most 25 m apart, shortlist nearby anchors, then rank distinct segments by exact projection distance. This approximate retrieval limits cost but can miss a reference arc. Six candidates are retained. Offset, distance above nearest, signed heading agreement, segment length, projected fraction, speed, log time gap and distance beyond an endpoint describe geometry and motion. Local arc IDs alone have no meaning across trips.")
    p("Training samples up to 100 evenly spaced fixes from each of 80 trips. Three group folds keep trips intact. All 19,160 fixes in the ten student-test trips are scored. Scalers fit inside training folds. The reserved evaluator partition was unavailable.")
    story.append(PageBreak());h("Model comparison")
    p("Choose the highest mean trip-grouped CV candidate macro F1. Each classifier predicts binary candidate relevance. At each fix, divide its six positive-class scores by their sum. Select the highest candidate for top-1 accuracy and convert that selection to six binary decisions for macro precision, recall and F1. Top-1 includes misses. Conditional log loss excludes observations whose reference arc is absent. The student-test miss rate was %.2f%%."%(100*m.miss_rate.iloc[0]))
    table([["Model","CV F1","Test F1","Top-1","Continuity","Log loss","Infer (s)"]]+[[n,*[f"{row[k]:.4f}" for k in ["cv_macro_f1","macro_f1","top1","continuity","log_loss","inference_s"]]] for n,(_,row) in zip(names,m.iterrows())])
    story.append(Image(str(out/"comparison.png"),width=495,height=178))
    p("Continuity is the equal-trip mean fraction of adjacent predictions that stay on one arc or follow a directly connected directed arc. Inference timing covers model scoring on the student-test features. The complete prediction request also loads files, constructs candidates, decodes and computes metrics.")
    sub("Interpreting the comparison")
    p("Boosting had the highest mean CV F1. Extra Trees narrowly led student-test top-1, but had a larger training-to-CV gap. Scores depend on the alignment convention and the sampled training fixes. See models.csv for fold scores, precision, recall and timing, and cm_*.csv for each 2-by-2 candidate confusion matrix (negative, positive order).")
    story.append(PageBreak());h("What each model learned")
    descriptions=[
      ("Logistic regression","Standardised inputs, C=1, L2 regularisation and logistic loss. A fast linear baseline tests whether offset and heading add enough evidence to separate matches. It cannot express strong feature interactions directly."),
      ("Gaussian Naive Bayes","Standardised inputs, variance smoothing 1e-9. Fits class-conditional Gaussian likelihoods with conditional independence. Distance and excess distance are correlated. Its highest log loss and lowest top-1 are consistent with poor probability quality."),
      ("Decision tree","Gini splits, maximum depth 6, at least 15 samples per leaf. Thresholds can combine heading and distance. A single tree makes abrupt probability changes around split boundaries."),
      ("Extra Trees","32 randomised trees, all features available at splits, minimum leaf size 6. Averages nonlinear partitions using Gini impurity. Its training-to-CV gap is the largest of the five, indicating some overfit despite good student-test top-1."),
      ("Histogram gradient boosting","60 iterations, up to 15 leaves and binary log loss. Sequential trees capture interactions. It won grouped CV and had the lowest student-test log loss. Inference was slower than the single tree.")]
    for i,(title,detail) in enumerate(descriptions):
        r=m.iloc[i];sub(title);p(detail+f" Training F1 {r.train_macro_f1:.4f}, CV F1 {r.cv_macro_f1:.4f}, student F1 {r.macro_f1:.4f}, top-1 {r.top1:.4f}, log loss {r.log_loss:.4f}.")
    sub("Bayesian optimisation")
    p(f"A Gaussian process with a Matérn kernel models grouped CV macro F1 over ordered leaf-count settings {spec['domain']}. Three initial evaluations cover the range. Expected improvement selects four subsequent trials. Leaf count controls interaction complexity. The best of seven trials returned {spec['value']} leaves, matching the untuned configuration. The search took {trials.seconds_elapsed.iloc[-1]:.2f} seconds in this run. Small timing differences between the identical configurations reflect run variability.")
    story.append(PageBreak());h("Route reconstruction")
    p("Hidden states are the six directed arc IDs at each fix. Log emissions use the normalised candidate scores. The transition log weights are 0 for staying, -0.45 for an immediately connected arc and -3.2 for other jumps, normalised over the available next candidates. These are heuristic defaults. Viterbi adds the emission to the best previous path score plus transition and backtracks. Unavailable candidates are excluded. A wholly empty row receives -1. A gap over 15 seconds or a non-increasing timestamp restarts decoding.")
    rows=[["Method","Point accuracy","Continuity","Sequence similarity"]]
    for k,g in dec.groupby("method"):
        rows.append([k,f"{np.average(g.accuracy,weights=g.fixes):.4f}",f"{g.continuity.mean():.4f}",f"{g.sequence_similarity.mean():.4f}"])
    table(rows)
    p("Sequence similarity is one minus edit distance divided by the longer sequence length, after collapsing consecutive repeats in aligned labels and predictions. It measures agreement with the observed aligned sequence and omits reference arcs skipped between GPS fixes. Sequence similarity and continuity are averaged equally across trips.")
    p("Trip 00000007 lost point accuracy (91.73% to 88.26%) despite improved continuity. The transition preference can keep predictions on the same road too long. Trip 00000000 improved from 92.73% to 94.81%. Direct adjacency also penalises legal paths across multiple unobserved segments.")
    sub("Student-test trip 00000007")
    story.append(Image(str(out/"geography.png"),width=490,height=234))
    p("Static geographical view of the same four data layers used in the interactive Folium map. Open results/development_map.html in the extracted project to toggle GPS, network, reference and predicted segments.")
    story.append(PageBreak());h("Distributed demonstration")
    p("Laptop 1 hosts FastAPI, the model, network and decoder. GET /trips lists the active partition. GET /predict/{trip} reconstructs the route from GPS and network inputs, then separately aligns any internal reference for scoring. The response contains the GPS trace, per-fix predicted IDs, predicted geometry and aggregate scores. Laptop 2 lists the trips and draws the selected route using the bundled Leaflet interface.")
    if (out/"api_checks.json").exists():
        checks=json.loads((out/"api_checks.json").read_text())
        table([["Development trip","Returned fixes","Decoded accuracy"]]+[[r["trip"],r["fixes"],f"{r['accuracy']:.3%}"] for r in checks])
    p("The reference run exercised HTTP endpoints locally on the student-test partition. The final physical two-laptop demo and reserved-trip scoring require the team's and organisers' hardware and data.")
    if (out/"ui.png").exists():story.append(Image(str(out/"ui.png"),width=475,height=292))
    sub("Evidence to add from the event")
    p("Insert a screenshot of Laptop 2 after selecting a trip and one of Laptop 1's receiving command window showing the successful request. Include the team's public repository URL, team identities, contributions, verified run details and AI-use disclosure.")
    doc=SimpleDocTemplate("PathSnap_Reference_Report.pdf",pagesize=A4,leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=42)
    doc.build(story)

if __name__=="__main__": main()
