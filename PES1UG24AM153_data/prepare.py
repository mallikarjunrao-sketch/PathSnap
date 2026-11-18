from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .trips import ROOT,load_trip,features,reference,inventory

def main():
    Path("results").mkdir(exist_ok=True)
    Path("cache").mkdir(exist_ok=True)
    stats = []
    for split in ("train","student_test"):
        packs = []
        for folder in sorted((ROOT/split).iterdir()):
            if not folder.is_dir(): continue
            t = load_trip(folder)
            stats.append(inventory(t))
            rows = np.arange(len(t.gps))
            if split=="train": # bound compute while representing every training trip
                rows = np.linspace(0,len(rows)-1,min(100,len(rows)),dtype=int)
            ids,X = features(t,rows)
            packs.append((folder.name,ids,X.astype("float32"),reference(t)[rows]))
        if not packs: raise ValueError("Empty partition: "+split)
        joblib.dump(packs,"cache/"+split+".joblib",compress=2)
    pd.DataFrame(stats).to_csv("results/inventory.csv",index=False)

if __name__=="__main__": main()
