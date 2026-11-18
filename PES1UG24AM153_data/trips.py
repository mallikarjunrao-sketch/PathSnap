"""Trip-local geometry. No reference file is opened by load_trip or features."""
from pathlib import Path
from types import SimpleNamespace
import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(os.environ.get("PATHSNAP_DATA", "data"))
K = 6
NAMES = ["offset_m", "candidate_gap_m", "heading_agreement", "segment_length_m",
         "fraction_along", "speed_mps", "log_time_gap", "off_end_m"]

def load_trip(folder):
    def read(ext, dtype):
        return np.loadtxt(folder / (folder.name + "." + ext), dtype=dtype, ndmin=2)
    gps, nodes, arcs = read("track", float), read("nodes", float), read("arcs", int)
    if not np.isfinite(gps).all() or not np.isfinite(nodes).all():
        raise ValueError(f"Non-finite coordinates in {folder}")
    if arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError("Invalid road node index")
    scale = np.array([111195 * np.cos(np.deg2rad(gps[:,1].mean())), 111195])
    origin = gps[0,:2]
    xy, net = (gps[:,:2]-origin)*scale, (nodes-origin)*scale
    a, b = net[arcs[:,0]], net[arcs[:,1]]
    vec = b-a
    length = np.linalg.norm(vec, axis=1)
    return SimpleNamespace(**locals())

def project(t, ids, xy):
    delta = xy[:,None,:] - t.a[ids]
    v = t.vec[ids]
    raw = (delta*v).sum(2)/np.maximum(t.length[ids]**2, 1e-12)
    f = raw.clip(0,1)
    dist = np.linalg.norm(delta-f[:,:,None]*v, axis=2)
    return dist, f, abs(raw-f)*t.length[ids]

def features(t, rows):
    # Anchor long segments at intervals of at most 25 metres for spatial retrieval.
    counts = np.maximum(1,np.ceil(t.length/25).astype(int))
    owners = np.repeat(np.arange(len(counts)), counts)
    fraction = (np.arange(len(owners))-np.repeat(np.cumsum(counts)-counts, counts)+0.5)/counts[owners]
    tree = cKDTree(t.a[owners]+fraction[:,None]*t.vec[owners])
    xy = t.xy[rows]
    width = min(48,len(owners))
    while True:
        near = owners[tree.query(xy,k=width)[1]].reshape(len(rows),width)
        dist,_,_ = project(t,near,xy)
        # First occurrence of each arc after sorting by ID.
        order = np.argsort(near,axis=1,kind="stable")
        ids = np.take_along_axis(near,order,1)
        d = np.take_along_axis(dist,order,1)
        d[:,1:] = np.where(np.diff(ids,axis=1)==0,np.inf,d[:,1:])
        if (np.isfinite(d).sum(1)>=K).all(): break
        if width==len(owners): raise ValueError("Trip has fewer than six candidate arcs")
        width = min(width*2,len(owners))
    ids = np.take_along_axis(ids,np.argsort(d,axis=1,kind="stable")[:,:K],1)
    d,f,off = project(t,ids,xy)
    prev = np.maximum(0,rows-1)
    motion = xy-t.xy[prev]
    step = np.linalg.norm(motion,axis=1)
    dt = t.gps[rows,2]-t.gps[prev,2]
    direction = (motion[:,None,:]*t.vec[ids]).sum(2)/np.maximum(step[:,None]*t.length[ids],1e-9)
    repeat = lambda v: np.broadcast_to(v[:,None],d.shape)
    X = np.stack([d,d-d.min(1,keepdims=True),direction,t.length[ids],f,
                  repeat(step/np.maximum(dt,1)),repeat(np.log1p(dt.clip(0))),off],axis=2)
    return ids,X

def reference(t):
    """Ordered route alignment for scoring or training only. These are proxy labels."""
    route = np.loadtxt(t.folder/(t.folder.name+".route"),dtype=int,ndmin=1)
    if route.min()<0 or route.max()>=len(t.arcs): raise ValueError("Invalid reference arc")
    # DP: align every fix to an ordered route position, allowing skipping and dwelling.
    # Absolute progress is a weak tie-breaker for overlapping/opposite-direction segments.
    lengths = t.length[route]
    mid = (lengths.cumsum()-lengths/2)/max(lengths.sum(),1)
    travelled = np.r_[0,np.linalg.norm(np.diff(t.xy,axis=0),axis=1)].cumsum()
    progress = travelled/max(travelled[-1],1)
    parents = np.empty((len(t.xy),len(route)),np.int32)
    cost = np.zeros(len(route))
    positions = np.arange(len(route))
    for start in range(0,len(t.xy),128):
        xy = t.xy[start:start+128]
        distances,_,_ = project(t,np.broadcast_to(route,(len(xy),len(route))),xy)
        for j,dist in enumerate(distances,start):
            lower = np.minimum.accumulate(cost)
            parents[j] = np.maximum.accumulate(np.where(cost<=lower,positions,0))
            cost = lower+dist+4*abs(mid-progress[j])
            cost -= cost.min()
    k = cost.argmin()
    result = np.empty(len(t.xy),int)
    for j in reversed(range(len(result))):
        result[j] = route[k]
        k = parents[j,k]
    return result

def inventory(t):
    graph = coo_matrix((np.ones(len(t.arcs)),t.arcs.T),shape=(len(t.nodes),len(t.nodes)))
    dt = np.diff(t.gps[:,2])
    return dict(split=t.folder.parent.name,trip=t.folder.name,fixes=len(t.gps),nodes=len(t.nodes),
        arcs=len(t.arcs),weak_components=connected_components(graph,directed=False,return_labels=False),
        isolated=len(t.nodes)-len(np.unique(t.arcs)),nonpositive_steps=int((dt<=0).sum()),
        one_second_steps=int((dt==1).sum()),max_gap_s=dt.max(),missing=int(np.isnan(t.gps).sum()),
        lon_min=t.gps[:,0].min(),lon_max=t.gps[:,0].max(),lat_min=t.gps[:,1].min(),
        lat_max=t.gps[:,1].max(),route_entries=len(np.loadtxt(t.folder/(t.folder.name+".route"),ndmin=1)),
        **{p.suffix[1:]+"_bytes":p.stat().st_size for p in t.folder.glob(t.folder.name+".*")})
