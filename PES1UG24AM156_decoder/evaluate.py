from pathlib import Path
import folium
import joblib
import numpy as np
import pandas as pd
from member1_data.trips import ROOT,load_trip
from member2_models.train import scores
from .decode import decode,continuity,sequence_score

def make_map(t,route):
    m=folium.Map(tiles=None,prefer_canvas=True)
    # Bundle Leaflet so the map works without internet.
    m.default_js=[("leaflet","../member4_demo/web/vendor/leaflet.min.js")]
    m.default_css=[("leaflet_css","../member4_demo/web/vendor/leaflet.css")]
    for label,ids,color,width in [
        ("Road network",np.arange(len(t.arcs)),"#abb4ab",1),
        ("Reference route",np.loadtxt(t.folder/(t.folder.name+".route"),int,ndmin=1),"#dc8229",7),
        ("Predicted route",np.unique(route),"#167555",3)]:
        layer=folium.FeatureGroup(name=label).add_to(m)
        folium.PolyLine(t.nodes[t.arcs[ids]][:,:,::-1].tolist(),color=color,weight=width).add_to(layer)
    points=t.gps[:,:2][:,::-1]
    layer=folium.FeatureGroup(name="GPS observations").add_to(m)
    for p in points: folium.CircleMarker(p,radius=1.5,color="#334aa0",weight=1).add_to(layer)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([points.min(0).tolist(),points.max(0).tolist()])
    m.save("results/development_map.html")

def main():
    model=joblib.load("artifacts/model.joblib")
    records=[]
    for name,ids,X,ref in joblib.load("cache/student_test.joblib"):
        t=load_trip(ROOT/"student_test"/name)
        prob=scores(model,X)
        raw=ids[np.arange(len(ref)),prob.argmax(1)]
        route=decode(t.arcs,ids,prob,t.gps[:,2])
        for label,seq in [("classifier",raw),("viterbi",route)]:
            records.append(dict(trip=name,method=label,fixes=len(ref),accuracy=np.mean(seq==ref),
                continuity=continuity(t.arcs,seq),sequence_similarity=sequence_score(ref,seq)))
        if name=="00000007": make_map(t,route)
    pd.DataFrame(records).to_csv("results/decoder.csv",index=False)

if __name__=="__main__": main()
