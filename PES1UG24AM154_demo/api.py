import logging
import os
import joblib
import numpy as np
from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from member1_data.trips import ROOT,load_trip,features,reference
from member2_models.train import scores
from member3_decoder.decode import decode,continuity

app = FastAPI(title="PathSnap")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["GET"])
model = joblib.load("artifacts/model.joblib")
partition = ROOT/os.environ.get("PATHSNAP_SPLIT","student_test")

@app.get("/trips")
def trips():
    return sorted(p.name for p in partition.iterdir() if p.is_dir() and (p/(p.name+".track")).is_file())

@app.get("/predict/{trip}")
def predict(trip:str):
    if trip not in trips(): raise HTTPException(404,"Unknown trip")
    t = load_trip(partition/trip)
    ids,X = features(t,np.arange(len(t.gps)))
    route = decode(t.arcs,ids,scores(model,X),t.gps[:,2])
    # Scoring boundary: inference is complete before this file is accessed.
    accuracy = None
    if (t.folder/(trip+".route")).is_file():
        accuracy = float(np.mean(reference(t)==route))
    changes = np.r_[True,np.diff(route)!=0]
    edges = t.nodes[t.arcs[route[changes]]][:,:,::-1]
    response = dict(trip=trip,fixes=len(route),accuracy=accuracy,continuity=float(continuity(t.arcs,route)),
        predicted_arc_ids=route.tolist(),segments=edges.tolist(),gps=t.gps[:,:2][:,::-1].tolist())
    logging.getLogger("uvicorn.error").info("Trip %s returned %d predictions, accuracy %s",trip,len(route),accuracy)
    return response
