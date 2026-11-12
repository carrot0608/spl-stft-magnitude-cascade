"""STFT-CNN healthy probability and magnitude-based conditional subtype fusion."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_cov, _class_means
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from torch.utils.data import DataLoader

from features import build_features
from model import STFTCNN, STFTDataset, set_seed

CLASS_NAMES = ('healthy', 'ischemia', 'simulated_hemorrhage')


def joint_scores(p_healthy, p_subtype):
    """Class order H/I/B; ties are resolved by the first class in this order."""
    h = np.asarray(p_healthy, dtype=np.float64)
    ib = np.asarray(p_subtype, dtype=np.float64)
    if h.ndim != 1 or ib.shape != (len(h),2):
        raise ValueError('Expected healthy (N,) and conditional I/B (N,2) probabilities.')
    if not (np.isfinite(h).all() and np.isfinite(ib).all()):
        raise ValueError('Nonfinite probabilities.')
    if np.any((h < 0) | (h > 1)) or np.any((ib < 0) | (ib > 1)) or not np.allclose(ib.sum(axis=1),1,atol=1e-5):
        raise ValueError('Invalid probability range or normalization.')
    return np.column_stack((h, (1-h)*ib[:,0], (1-h)*ib[:,1]))


class ShrinkageLDA(LinearDiscriminantAnalysis):
    def _solve_lstsq(self, X, y, shrinkage, covariance_estimator):
        self.means_ = _class_means(X,y)
        self.covariance_ = _class_cov(X,y,self.priors_,shrinkage,covariance_estimator)
        try:
            coefficients = linalg.cho_solve(linalg.cho_factor(self.covariance_,lower=True),self.means_.T)
        except linalg.LinAlgError:
            coefficients = linalg.lstsq(self.covariance_,self.means_.T)[0]
        self.coef_ = coefficients.T
        self.intercept_ = -0.5*np.diag(self.means_@self.coef_.T)+np.log(self.priors_)


def _loader(x, y, training=False):
    return DataLoader(STFTDataset(x,y,training), batch_size=4, shuffle=training,
                      pin_memory=bool(torch.cuda.is_available()), num_workers=0)


def _probabilities(model, loader, device):
    model.eval()
    parts = []
    with torch.no_grad():
        for x, _ in loader:
            with torch.cuda.amp.autocast(enabled=str(device).startswith('cuda')):
                logits = model(x.to(device))
            parts.append(torch.softmax(logits.float(),dim=1).cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def _train_member(x, y, tr, va, seed, device, epochs=80):
    set_seed(seed)
    model = STFTCNN().to(device)
    binary = (y != 0).astype(np.int64)
    counts = np.bincount(binary[tr],minlength=2).astype(np.float32)
    if np.any(counts == 0):
        raise ValueError('Each inner training fold must include healthy and pathological states.')
    weights = counts.sum()/(2*counts)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    train = _loader(x[tr],binary[tr],True)
    validation = _loader(x[va],binary[va])
    amp = str(device).startswith('cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    history, states = [], []
    for _ in range(epochs):
        model.train()
        for xb, yb in train:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                loss = loss_fn(model(xb.to(device)),yb.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss.')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        history.append(_probabilities(model,validation,device))
        states.append({k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    return np.stack(history), states


def fit_subtype(x, labels, seed):
    if set(labels.tolist()) != {0,1}:
        raise ValueError('Subtype training requires both ischemia and simulated hemorrhage.')
    models = [
        Pipeline([('scale',StandardScaler()),('model',LogisticRegression(C=1,class_weight='balanced',max_iter=5000,random_state=seed))]),
        Pipeline([('scale',StandardScaler()),('model',ShrinkageLDA(solver='lsqr',shrinkage='auto'))]),
        Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=max(1,min(10,len(x)-2,x.shape[1])),random_state=seed+2000000)),
                  ('model',SVC(C=1,kernel='linear',gamma='scale',class_weight='balanced',probability=True,random_state=seed+2000000))])]
    with threadpool_limits(limits=1):
        for model in models:
            model.fit(x,labels)
            if list(model.classes_) != [0,1]:
                raise ValueError('Unexpected subtype class order.')
    return models


class SoftFusionClassifier:
    """Fit on training animals only. Test features and labels are not fit arguments."""
    def fit(self, spectra, labels, animal_ids, seed=42, outer_fold=1, device='cuda'):
        torch.set_num_threads(4)
        x, magnitude = build_features(spectra)
        y, groups = np.asarray(labels), np.asarray(animal_ids)
        if y.shape != (len(x),) or groups.shape != y.shape or set(y.tolist()) != {0,1,2}:
            raise ValueError('Aligned labels 0/1/2 and animal IDs are required.')
        if len(set(groups)) < 3:
            raise ValueError('At least three training animals are required.')
        self.device = device
        oof = np.full((80,len(y),2),np.nan,dtype=np.float32)
        self.models = []
        with TemporaryDirectory(prefix='soft_fusion_') as scratch:
            members = []
            for inner,(tr,va) in enumerate(LeaveOneGroupOut().split(x,y,groups),1):
                if set(groups[tr]) & set(groups[va]):
                    raise RuntimeError('Animal overlap.')
                predictions = []
                for member in range(5):
                    member_seed = seed+outer_fold*100000+inner*1000+member*10000000
                    history, states = _train_member(x,y,tr,va,member_seed,device)
                    path = Path(scratch)/f'{inner}_{member}.pt'
                    torch.save(states,path)
                    members.append((inner,path))
                    predictions.append(history)
                    del states
                oof[:,va] = np.mean(np.stack(predictions),axis=0)
            binary = (y != 0).astype(int)
            ce = [-np.log(np.clip(p[np.arange(len(y)),binary],1e-8,1)).mean() for p in oof]
            self.epoch = int(np.argmin(ce))+1
            for inner,path in members:
                model = STFTCNN()
                states = torch.load(path,map_location='cpu')
                model.load_state_dict(states[self.epoch-1])
                self.models.append((inner,model))
                del states
        pathologies = y != 0
        self.subtype = fit_subtype(magnitude[pathologies],y[pathologies]-1,seed+outer_fold*100000+5000000)
        return self

    def predict_proba(self, spectra):
        x, magnitude = build_features(spectra)
        loader = _loader(x,np.zeros(len(x),dtype=int))
        inner_ids = sorted({inner for inner,_ in self.models})
        sums = np.zeros((len(x),2),dtype=np.float64)
        for inner in inner_ids:
            parts = []
            for group,model in self.models:
                if group == inner:
                    model.to(self.device)
                    parts.append(_probabilities(model,loader,self.device))
                    model.cpu()
            sums += np.mean(np.stack(parts),axis=0)
        p1 = (sums/len(inner_ids)).astype(np.float32)
        with threadpool_limits(limits=1):
            p2 = np.mean(np.stack([model.predict_proba(magnitude).astype(np.float64) for model in self.subtype]),axis=0).astype(np.float32)
        return joint_scores(p1[:,0],p2)

    def predict(self, spectra):
        return self.predict_proba(spectra).argmax(axis=1)
