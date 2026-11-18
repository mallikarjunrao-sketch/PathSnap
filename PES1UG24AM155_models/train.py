from pathlib import Path
from time import perf_counter
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import precision_recall_fscore_support,log_loss,confusion_matrix
from member1_data.trips import ROOT,K
from member3_decoder.decode import continuity

MODELS = {
    "logistic": (make_pipeline(StandardScaler(),LogisticRegression(max_iter=300)), "logisticregression__C",[.03,.1,.3,1,3,10,30,100]),
    "gaussian_nb": (make_pipeline(StandardScaler(),GaussianNB()),"gaussiannb__var_smoothing",np.logspace(-11,-2,10)),
    "tree": (DecisionTreeClassifier(max_depth=6,min_samples_leaf=15,random_state=42),"max_depth",list(range(2,14))),
    "extra_trees": (ExtraTreesClassifier(n_estimators=32,min_samples_leaf=6,max_features=1.,random_state=42,n_jobs=1),"min_samples_leaf",[1,2,3,4,6,8,12,16,24,32]),
    "hist_boost": (HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=15,early_stopping=False,random_state=42),"max_leaf_nodes",[3,5,7,10,15,20,25,31,42,56])}

def scores(model,X):
    p = model.predict_proba(X.reshape(-1,X.shape[-1]))[:,1].reshape(X.shape[:2]).clip(1e-9,1-1e-9)
    return p/p.sum(1,keepdims=True)

def arrays(packs):
    X = np.concatenate([x for _,_,x,_ in packs])
    matches = np.concatenate([ids==ref[:,None] for _,ids,_,ref in packs])
    groups = np.concatenate([np.repeat(name,len(ref)) for name,_,_,ref in packs])
    return X,matches,groups

def fit(m,X,y):
    return m.fit(X.reshape(-1,X.shape[-1]),y.ravel())

def classification(matches,prob):
    # Candidate-level metrics after selecting exactly one arc per observation.
    pred = np.eye(K,dtype=bool)[prob.argmax(1)]
    y,h = matches.ravel(),pred.ravel()
    precision,recall,f1,_ = precision_recall_fscore_support(y,h,average="macro",zero_division=0)
    hit = (matches*pred).any(1)
    present = matches.any(1)
    pos = (matches*prob).sum(1)
    return dict(macro_f1=f1,macro_precision=precision,macro_recall=recall,top1=hit.mean(),
        miss_rate=1-present.mean(),log_loss=-np.log(pos[present].clip(1e-12)).mean(),
        confusion=confusion_matrix(y,h,labels=[False,True]).tolist())

def main():
    train = joblib.load("cache/train.joblib")
    test = joblib.load("cache/student_test.joblib")
    X,y,g = arrays(train)
    Xt,yt,_ = arrays(test)
    folds = list(GroupKFold(3).split(X,groups=g))
    def cv(m):
        return [classification(y[b],scores(fit(clone(m),X[a],y[a]),X[b]))["macro_f1"] for a,b in folds]
    def experiment(name,m):
        start = perf_counter()
        print(f"Training {name} with three trip-grouped folds and a final refit...", flush=True)
        values = cv(m)
        fit(m,X,y)
        training = perf_counter()-start
        start = perf_counter()
        prob = scores(m,Xt)
        timing = perf_counter()-start
        result = classification(yt,prob)
        print(f"  Fold F1: {values} | Mean: {np.mean(values):.4f} | Student top-1: {result['top1']:.2%}", flush=True)
        confusion = result.pop("confusion")
        np.savetxt("results/cm_"+name+".csv",confusion,fmt="%d",delimiter=",")
        chunks = np.split(prob,np.cumsum([len(ref) for _,_,_,ref in test])[:-1])
        cont = []
        for (trip,ids,_,ref),p in zip(test,chunks):
            arcs = np.loadtxt(ROOT/"student_test"/trip/(trip+".arcs"),dtype=int)
            cont.append(continuity(arcs,ids[np.arange(len(ref)),p.argmax(1)]))
        return dict(model=name,cv_macro_f1=np.mean(values),folds=str(values),**result,
            train_macro_f1=classification(y,scores(m,X))["macro_f1"],continuity=np.mean(cont),
            train_cv_s=training,inference_s=timing,config=str(m))
    results = []
    for name,(m,_,_) in MODELS.items():
        results.append(experiment(name,m))
        pd.DataFrame(results).to_csv("results/models.csv",index=False)
    best = max(results,key=lambda r:r["cv_macro_f1"])["model"]
    base,param,grid = MODELS[best]
    # Sequential GP Bayesian optimisation over an ordered discrete domain.
    print(f"Selected {best}. Bayesian search over {param}: {grid}", flush=True)
    axis = np.linspace(0,1,len(grid))[:,None]
    seen,values,trials = [],[],[]
    start = perf_counter()
    for iteration in range(7):
        if iteration<3: idx=[0,len(grid)//2,len(grid)-1][iteration]
        else:
            gp = GaussianProcessRegressor(kernel=Matern(.3,nu=2.5),alpha=.01,normalize_y=True,optimizer=None)
            gp.fit(axis[seen],values)
            mu,sigma = gp.predict(axis,return_std=True)
            gain = mu-max(values)-.0001
            z = gain/np.maximum(sigma,1e-12)
            ei = gain*norm.cdf(z)+sigma*norm.pdf(z)
            ei[seen] = -np.inf
            idx = int(ei.argmax())
        value = grid[idx]
        measure = np.mean(cv(clone(base).set_params(**{param:value})))
        print(f"Trial {iteration+1}/7: {param}={value}, grouped CV macro F1={measure:.4f}", flush=True)
        seen.append(idx)
        values.append(measure)
        trials.append(dict(trial=iteration+1,model=best,parameter=param,value=value,cv_macro_f1=measure,
            seconds_elapsed=perf_counter()-start))
        pd.DataFrame(trials).to_csv("results/bayesian_trials.csv",index=False)
    tuned = clone(base).set_params(**{param:grid[seen[np.argmax(values)]]})
    results.append(experiment(best+"_tuned",tuned))
    pd.DataFrame(results).to_csv("results/models.csv",index=False)
    joblib.dump(tuned,"artifacts/model.joblib",compress=3)
    Path("artifacts/selection.json").write_text(json.dumps({"winner":best,"parameter":param,
        "domain":list(grid),"value":grid[seen[np.argmax(values)]],"seed":42},indent=2))

if __name__=="__main__": main()
