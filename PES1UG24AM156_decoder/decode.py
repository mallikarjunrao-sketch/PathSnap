"""Log-space Viterbi with a soft penalty for skipped or disconnected arcs."""
import numpy as np

CONFIG = {"connected_cost":0.45,"jump_cost":3.2,"gap_seconds":15}

def continuity(arcs, route):
    previous,current = route[:-1],route[1:]
    return np.mean((previous==current)|(arcs[previous,1]==arcs[current,0])) if len(previous) else 1.

def decode(arcs, ids, prob, times, config=CONFIG):
    if len(ids)==0: return np.empty(0,int)
    valid = (ids>=0)&np.isfinite(prob)&(prob>0)
    emission = np.log(np.where(valid,prob,1.))
    emission[~valid] = -np.inf
    output = np.full(len(ids),-1,int)
    cuts = np.flatnonzero((np.diff(times)>config["gap_seconds"])|(np.diff(times)<=0))+1
    empty = np.flatnonzero(~valid.any(1))
    boundaries = np.unique(np.r_[0,cuts,empty,empty+1,len(ids)])
    for lo,hi in zip(boundaries[:-1],boundaries[1:]):
        if not valid[lo].any(): continue
        s = ids[lo:hi].clip(0)
        same = s[:-1,:,None]==s[1:,None,:]
        linked = arcs[s[:-1],1][:,:,None]==arcs[s[1:],0][:,None,:]
        logtrans = -np.where(same,0.,np.where(linked,config["connected_cost"],config["jump_cost"]))
        logtrans = np.where(valid[lo+1:hi,None,:],logtrans,-np.inf)
        logtrans -= np.logaddexp.reduce(logtrans,axis=2,keepdims=True)
        score = emission[lo].copy()
        parents = np.zeros(s.shape,int)
        for j in range(1,len(s)):
            pair = score[:,None]+logtrans[j-1]
            parents[j] = pair.argmax(0)
            score = pair.max(0)+emission[lo+j]
        state = score.argmax()
        for j in reversed(range(len(s))):
            output[lo+j] = s[j,state]
            state = parents[j,state]
    return output

def sequence_score(reference,predicted):
    """Normalised edit similarity after collapsing consecutive repeats."""
    def collapse(x): return x[np.r_[True,np.diff(x)!=0]]
    a,b = collapse(reference),collapse(predicted)
    previous = list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current = [i]
        for j,y in enumerate(b,1):
            current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+int(x!=y)))
        previous = current
    return 1-previous[-1]/max(len(a),len(b))
