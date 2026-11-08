import os,copy,itertools,json,pickle,time
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import torch
from torch import nn
from scipy.optimize import linprog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from pypower.api import case14,case30,case57,case118
exec(open('work/analysis.py').read().split('def run():')[0])
torch.set_num_threads(1)
KINDS=['missing','x10','x0.1','r10','b0','b2']
FNS={14:case14,30:case30,57:case57,118:case118}

def network(n):
 d=FNS[n]();bus=d['bus'];n=len(bus);slack=int(np.where(bus[:,1]==3)[0][0]);order=[slack]+[i for i in range(n) if i!=slack];mp={int(bus[i,0]):j+1 for j,i in enumerate(order)};bb=bus[order];br=d['branch'];br=br[br[:,10]>0];arr=np.array([[mp[int(z[0])],mp[int(z[1])],z[2],z[3],z[4],z[8] if z[8] else 1] for z in br]);pg=np.zeros(n);vs=bb[:,7].copy()
 for g in d['gen']:
  if g[7]>0:idx=mp[int(g[0])]-1;pg[idx]+=g[1]/d['baseMVA'];vs[idx]=g[5]
 return dict(n=n,br=arr,pv=list(np.where(bb[:,1]==2)[0]),vs=vs,pd=bb[:,2]/d['baseMVA'],qd=bb[:,3]/d['baseMVA'],pg=pg,sh=bb[:,5]/d['baseMVA'])

def fastscreen(c,x,e):
 n=c['n'];Y=Ybus(c);a,q=indices(c);J=jac(c,x,Y);b1=abs(J)@e;ev=e[n:];et=e[:n];v=x[n:]+ev;hh=et[:,None]+et[None,:];A=abs(Y);np.fill_diagonal(A,0)
 rem=(A*(.5*v[:,None]*v[None,:]*hh**2+hh*(v[None,:]*ev[:,None]+v[:,None]*ev[None,:])+ev[:,None]*ev[None,:])).sum(axis=1)
 R=np.r_[abs(Y.diagonal().real[a])*ev[a]**2+rem[a],abs(Y.diagonal().imag[q])*ev[q]**2+rem[q]];beta=b1+R;f=F(c,x,Y);lo=np.maximum(0,abs(f)-beta)
 return f,J,R,beta,lo

def inventory(c,seed):
 rng=np.random.default_rng(seed);n=c['n'];present={tuple(sorted((int(z[0])-1,int(z[1])-1))) for z in c['br']};adj=np.zeros((n,n),int)
 for i,j in present:adj[i,j]=adj[j,i]=1
 paths=adj@adj;possible=[(i,j) for i in range(n) for j in range(i+1,n) if (i,j) not in present and paths[i,j]>0];rng.shuffle(possible)
 edges=list(np.sort(c['br'][:,:2].astype(int)-1,axis=1))+sorted(possible[:max(5,len(present)//4)]);return np.array(edges,int)

def mutate(c,j,k):
 d=copy.deepcopy(c);r,x,b=c['br'][j,2:5]
 if k=='missing':
  d['br']=np.delete(d['br'],j,0);d['ids']=np.delete(np.arange(len(c['br'])),j)
 else:d['ids']=np.arange(len(c['br']))
 if k=='missing':pass
 elif k=='x10':d['br'][j,3]*=10
 elif k=='x0.1':d['br'][j,3]*=.1
 elif k=='r10':d['br'][j,2]*=10
 elif k=='b0':d['br'][j,4]=0
 else:d['br'][j,4]*=2
 return d

def features(c,x,e,edges,physics=True):
 n=c['n'];f,J,R,beta,lo=fastscreen(c,x,e);a,q=indices(c);fp=np.zeros(n);fq=np.zeros(n);bp=np.ones(n);bq=np.ones(n);lp=np.zeros(n);lq=np.zeros(n)
 fp[a]=f[:len(a)];fq[q]=f[len(a):];bp[a]=beta[:len(a)];bq[q]=beta[len(a):];lp[a]=lo[:len(a)];lq[q]=lo[len(a):]
 typ=np.zeros((n,3));typ[0,0]=1;typ[c['pv'],1]=1;typ[q,2]=1
 sx=lambda v:np.sign(v)*np.log1p(abs(v)/.001)
 node=np.c_[x[n:]-1,x[:n],c['pg']-c['pd'],-c['qd'],typ,sx(fp),sx(fq),np.log1p(bp/.001),np.log1p(bq/.001),np.log1p(lp/.001),np.log1p(lq/.001),np.full(n,np.log10(e[n])),np.full(n,np.log10(e[0]))]
 pairs={int(j):z[2:] for j,z in zip(c.get('ids',np.arange(len(c['br']))),c['br'])};edge=[]
 for slot,(i,j) in enumerate(edges):
  p=pairs.get(slot);present=int(p is not None);r,xv,b,t=p if present else [0,0,0,1]
  edge.append([present,np.log1p(r*10),np.log1p(xv*10),b,t-1,abs(x[n+i]-x[n+j]),abs(x[i]-x[j])])
 edge=np.array(edge);ii,jj=edges.T
 # shared edge feature vector used by all tabular baselines and neural decoder
 tab=np.c_[node[ii]+node[jj],abs(node[ii]-node[jj]),edge]
 raw=abs(fp[ii])+abs(fp[jj])+abs(fq[ii])+abs(fq[jj])
 return node.astype('float32'),edge.astype('float32'),tab.astype('float32'),raw.astype('float32'),float(max(lo))

def generate():
 rng=np.random.default_rng(4501);data={};manifest={}
 for n in [14,30,57,118]:
  base=network(n);edges=inventory(base,n+100);N=len(base['br']);idx=rng.permutation(N);cut=int(.7*N);vcut=int(.85*N);groups={'train':idx[:cut],'val':idx[cut:vcut],'test':idx[vcut:]}
  if n==118:groups={'transfer':idx}
  manifest[n]={'buses':n,'branches':N,'inventory':len(edges),'fault_branches':{k:sorted(map(int,v)) for k,v in groups.items()}}
  for split,allowed in groups.items():
   count={'train':700,'val':160,'test':220,'transfer':350}[split];samples=[];attempt=0
   while len(samples)<count:
    attempt+=1;c=copy.deepcopy(base);factors=rng.uniform(.85,1.15,n);c['pd']*=factors;c['qd']*=factors
    if split in ['test','transfer']:c['pd']*=1.1;c['qd']*=1.1
    try:truth,_=solve(c)
    except Exception:continue
    j=int(rng.choice(allowed));kind=str(rng.choice(KINDS));row=c['br'][j]
    if (kind=='r10' and row[2]==0) or (kind in ['b0','b2'] and row[4]==0):continue
    # charging at two regulated buses is structurally unobservable; exclude from supervised truth
    touched=[int(row[0])-1,int(row[1])-1]
    if kind in ['b0','b2'] and all(i==0 or i in c['pv'] for i in touched):continue
    d=mutate(c,j,kind);p=int(rng.choice([2,3,4,5]));xp,e=rounded(truth,p,p-1);node,edge,tab,raw,lower=features(d,xp,e,edges)
    target=j;samples.append(dict(node=node,edge=edge,tab=tab,raw=raw,target=target,kind=KINDS.index(kind),p=p,n=n,edges=edges,lower=lower,j=j,branch=d['br'],record_ids=d['ids'].copy(),pd=d['pd'],qd=d['qd'],pg=d['pg'],vs=d['vs'],pv=d['pv'],sh=d['sh'],state=xp,radii=e))
   data[(n,split)]=samples;print('generated',n,split,len(samples),'attempts',attempt,flush=True)
 with open('learning/data.pkl','wb') as f:pickle.dump((data,manifest),f,protocol=4)
 json.dump(manifest,open('learning/manifest.json','w'),indent=2)

class RankGNN(nn.Module):
 def __init__(self,F,E,kind='edge',physics=True):
  super().__init__();self.kind=kind;self.physics=physics;h=32
  self.enc=nn.Linear(F,h);self.selfs=nn.ModuleList([nn.Linear(h,h) for _ in range(2)]);self.msgs=nn.ModuleList([nn.Linear(h,h,bias=False) for _ in range(2)]);self.gates=nn.ModuleList([nn.Linear(E,h) for _ in range(2)]);self.decode=nn.Sequential(nn.Linear(2*h+E+2*F,48),nn.ReLU());self.loc=nn.Linear(48,1);self.cls=nn.Linear(48,6)
 def forward(self,node,edge,idx):
  if not self.physics:
   node=node.clone();node[:,:,7:13]=0
  h=torch.relu(self.enc(node));i,j=idx[:,0],idx[:,1]
  if self.kind!='mlp':
   for W,M,G in zip(self.selfs,self.msgs,self.gates):
    gate=torch.sigmoid(G(edge)) if self.kind=='edge' else torch.ones_like(h[:,i])
    # absent candidate edges do not carry messages, but remain candidates for prediction
    gate=gate*edge[:,:,:1];mi=M(h[:,j])*gate;mj=M(h[:,i])*gate
    agg=torch.zeros_like(h);agg.index_add_(1,i,mi);agg.index_add_(1,j,mj)
    deg=torch.zeros((h.shape[0],h.shape[1],1),dtype=edge.dtype,device=edge.device);deg.index_add_(1,i,edge[:,:,:1]);deg.index_add_(1,j,edge[:,:,:1]);h=torch.relu(W(h)+agg/deg.clamp_min(1))
  cat=torch.cat([h[:,i]+h[:,j],abs(h[:,i]-h[:,j]),edge,node[:,i]+node[:,j],abs(node[:,i]-node[:,j])],dim=-1)
  z=self.decode(cat);return self.loc(z).squeeze(-1),self.cls(z)

def pack(samples):
 return torch.tensor(np.array([x['node'] for x in samples])),torch.tensor(np.array([x['edge'] for x in samples])),torch.tensor(samples[0]['edges']),torch.tensor([x['target'] for x in samples])
def evaluate(net,samples,classes=False):
 scores=[];types=[]
 with torch.no_grad():
  for k in range(0,len(samples),64):
   a,b,ii,y=pack(samples[k:k+64]);loc,cl=net(a,b,ii);scores.extend(loc.numpy());types.extend(cl.numpy())
 return (np.array(scores),np.array(types)) if classes else np.array(scores)
def metrics(scores,samples):
 rank=np.array([int(np.where(np.argsort(-s,kind='stable')==x['target'])[0][0])+1 for s,x in zip(scores,samples)]);return {'top1':float(np.mean(rank==1)),'top3':float(np.mean(rank<=3)),'top5':float(np.mean(rank<=5)),'mrr':float(np.mean(1/rank)),'mean_rank':float(np.mean(rank)),'rank':rank.tolist()}

def train_all():
 data,manifest=pickle.load(open('learning/data.pkl','rb'));train=sum([data[n,'train'] for n in [14,30,57]],[]);val=sum([data[n,'val'] for n in [14,30,57]],[]);result={'manifest':manifest,'models':{},'curves':{}}
 # train normalization only; keep raw presence indicator for message adjacency
 A=np.concatenate([x['node'] for x in train]);mu=A.mean(0);sd=A.std(0).clip(.05);Bedge=np.concatenate([x['edge'] for x in train]);em=Bedge.mean(0);es=Bedge.std(0).clip(.05);em[0]=0;es[0]=1
 for ss in data.values():
  for x in ss:
   x['node']=(x['node']-mu)/sd;x['edge']=(x['edge']-em)/es;i,j=x['edges'].T;x['tab']=np.c_[x['node'][i]+x['node'][j],abs(x['node'][i]-x['node'][j]),x['edge']]
 Fdim=train[0]['node'].shape[1];Edim=train[0]['edge'].shape[1]
 rng=np.random.default_rng(29)
 for name,kind,phy in [('EdgeGNN','edge',True),('GraphSAGE','mean',True),('EdgeMLP','mlp',True),('NoPhysics','edge',False)]:
  torch.manual_seed(29);net=RankGNN(Fdim,Edim,kind,phy);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);best=-1;beststate=None;curve=[]
  for epoch in range(55):
   net.train();losses=[]
   for n in rng.permutation([14,30,57]):
    ss=data[int(n),'train'];order=rng.permutation(len(ss))
    for k in range(0,len(ss),32):
     batch=[ss[i] for i in order[k:k+32]];a,b,ii,y=pack(batch);opt.zero_grad();loc,cl=net(a,b,ii);loss=nn.functional.cross_entropy(loc,y)+.3*nn.functional.cross_entropy(cl[torch.arange(len(batch)),y],torch.tensor([z['kind'] for z in batch]));loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(float(loss.detach()))
   net.eval();va=[]
   for n in [14,30,57]:va.extend(metrics(evaluate(net,data[n,'val']),data[n,'val'])['rank'])
   score=float(np.mean(1/np.array(va)));curve.append([epoch,float(np.mean(losses)),score])
   if score>best:best=score;beststate=copy.deepcopy(net.state_dict())
  net.load_state_dict(beststate);result['models'][name]={};result['curves'][name]=curve
  for n,split in [(14,'test'),(30,'test'),(57,'test'),(118,'transfer')]:
   ss=data[n,split];s,cl=evaluate(net,ss,True);met=metrics(s,ss);pred=np.argmax(s,axis=1);ct=cl[np.arange(len(ss)),pred].argmax(1);met['joint_top1']=float(np.mean([(pi==z['target'] and ki==z['kind']) for pi,ki,z in zip(pred,ct,ss)]));met['type_at_true_edge']=float(np.mean(cl[np.arange(len(ss)),[z['target'] for z in ss]].argmax(1)==[z['kind'] for z in ss]));result['models'][name][str(n)]=met
   for x,pred in zip(ss,s):x.setdefault('scores',{})[name]=pred
  torch.save(dict(state=beststate,mu=mu,sd=sd,em=em,es=es,kind=kind,physics=phy),f'learning/{name}.pt')
  print(name,{n:round(z['top1'],3) for n,z in result['models'][name].items()},flush=True)
 # RF with per-edge binary labels, same information no graph message passing
 XX=np.concatenate([x['tab'] for x in train]);yy=np.concatenate([np.arange(len(x['tab']))==x['target'] for x in train]);rf=RandomForestClassifier(n_estimators=120,max_depth=16,min_samples_leaf=2,class_weight='balanced_subsample',random_state=29,n_jobs=1);rf.fit(XX,yy)
 for name in ['RandomForest','Residual','Random']:
  result['models'][name]={}
  for n,split in [(14,'test'),(30,'test'),(57,'test'),(118,'transfer')]:
   ss=data[n,split];sc=[]
   for x in ss:
    pred=rf.predict_proba(x['tab'])[:,1] if name=='RandomForest' else (x['raw'] if name=='Residual' else rng.uniform(size=len(x['edge'])))
    sc.append(pred);x.setdefault('scores',{})[name]=pred
   result['models'][name][str(n)]=metrics(sc,ss)
  print(name,{n:round(z['top1'],3) for n,z in result['models'][name].items()},flush=True)
 # precision and class breakdown for EdgeGNN vs RF and residual
 result['breakdown']={}
 for label,field,values in [('precision','p',[2,3,4,5]),('kind','kind',list(range(6)))]:
  result['breakdown'][label]={}
  for v in values:
   subset=sum([[x for x in data[n,'test'] if x[field]==v] for n in [14,30,57]],[]);result['breakdown'][label][str(v)]={'N':len(subset)}
   for name in ['EdgeGNN','RandomForest','Residual']:
    result['breakdown'][label][str(v)][name]=metrics([x['scores'][name] for x in subset],subset)
 json.dump(result,open('learning/results.json','w'),indent=2)
 with open('learning/evaluated.pkl','wb') as f:pickle.dump((data,result),f,protocol=4)
if __name__=='__main__':
 if not os.path.exists('learning/data.pkl'):generate()
 train_all()
