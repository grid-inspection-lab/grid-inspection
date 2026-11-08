exec(open('learning/study.py').read().split("if __name__=='__main__':")[0])
from sklearn.metrics import confusion_matrix
raw,man=pickle.load(open('learning/data.pkl','rb'));data,res=pickle.load(open('learning/evaluated.pkl','rb'));ck=torch.load('learning/EdgeGNN.pt',weights_only=False);net=RankGNN(15,7,'edge',True);net.load_state_dict(ck['state']);net.eval();audit={}
# graph relabeling and edge orientation/ordering checks
x=data[14,'test'][0];a,b,ii,y=pack([x]);base=net(a,b,ii)[0].detach().numpy();rng=np.random.default_rng(9);perm=rng.permutation(14);inv=np.argsort(perm);score=net(a[:,perm],b,torch.tensor(inv[ii.numpy()]))[0].detach().numpy();audit['node_permutation_error']=float(np.max(abs(score-base)))
ep=rng.permutation(len(ii));score=net(a,b[:,ep],ii[ep]) [0].detach().numpy()[:,np.argsort(ep)];audit['edge_permutation_error']=float(np.max(abs(score-base)))
score=net(a,b,ii[:,[1,0]])[0].detach().numpy();audit['edge_orientation_error']=float(np.max(abs(score-base)))
assert max(audit.values())<2e-5
# directional derivative on a small actual graph, finite differences independent of backprop
sc=evaluate(net,data[14,'test']);hard=int(np.argmax(np.log(np.exp(sc-sc.max(1,keepdims=True)).sum(1))+sc.max(1)-sc[np.arange(len(sc)),[z['target'] for z in data[14,'test']]]));a,b,ii,y=pack([data[14,'test'][hard]]);torch.manual_seed(901);nd=copy.deepcopy(net).double();aa=a.double().requires_grad_();bb=b.double();loc,cl=nd(aa,bb,ii);loss=nn.functional.cross_entropy(loc,y);loss.backward();direction=torch.randn_like(aa);ad=float((aa.grad*direction).sum());eps=1e-5
with torch.no_grad():fd=float((nn.functional.cross_entropy(nd(aa+eps*direction,bb,ii)[0],y)-nn.functional.cross_entropy(nd(aa-eps*direction,bb,ii)[0],y))/(2*eps))
audit['gradient_autodiff']=ad;audit['gradient_finite_difference']=fd;audit['gradient_relative_error']=abs(ad-fd)/max(1,abs(ad));assert audit['gradient_relative_error']<1e-6
# observed prediction confidence and conditional fault-type confusion
reliability={};classes={};all_scores={};verifier={}
for n,split in [(14,'test'),(30,'test'),(57,'test'),(118,'transfer')]:
 ss=data[n,split];scores,types=evaluate(net,ss,True);p=np.exp(scores-scores.max(1,keepdims=True));p/=p.sum(1,keepdims=True);pred=p.argmax(1);conf=p.max(1);correct=pred==[z['target'] for z in ss]
 bins=[];ece=0
 for lo in np.linspace(0,.9,10):
  ix=(conf>=lo)&(conf<(lo+.1) if lo<.9 else conf<=1)
  if sum(ix):bins.append([float(lo+.05),int(sum(ix)),float(conf[ix].mean()),float(correct[ix].mean())]);ece+=sum(ix)/len(ss)*abs(float(conf[ix].mean()-correct[ix].mean()))
 order=np.argsort(-conf);coverage=np.arange(1,len(ss)+1)/len(ss);risk=1-np.cumsum(correct[order])/np.arange(1,len(ss)+1)
 reliability[str(n)]={'bins':bins,'ece':ece,'coverage':coverage.tolist(),'risk':risk.tolist(),'confidence':conf.tolist(),'correct':correct.tolist()}
 cls=types[np.arange(len(ss)),[z['target'] for z in ss]].argmax(1);classes[str(n)]=confusion_matrix([z['kind'] for z in ss],cls,labels=range(6)).tolist()
 for z,s in zip(ss,p):z['probabilities']=s
 # source-repaired truth must admit an actual point within the printed box
 passed=0;maxres=0
 for z in raw[n,split]:
  c=network(n)
  for key in ['pd','qd','pg','vs','pv','sh']:c[key]=z[key]
  truth,_=solve(c);f=F(c,truth);mx=float(max(abs(f)));maxres=max(maxres,mx)
  passed+=int(np.all(abs(truth-z['state'])<=z['radii']+1e-11) and mx<1e-10)
 verifier[str(n)]={'passed':passed,'N':len(ss),'max_residual':maxres}
audit.update(reliability=reliability,conditional_type_confusion=classes,source_repair=verifier)
# branch-group bootstrap, preserving the concentration of examples on withheld circuits
boot={};gen=np.random.default_rng(800)
for n in [14,30,57,118]:
 split='transfer' if n==118 else 'test';ss=data[n,split];ids=sorted(set(z['j'] for z in ss));dif=[]
 for _ in range(1000):
  chosen=gen.choice(ids,len(ids),replace=True);samples=[k for j in chosen for k,z in enumerate(ss) if z['j']==j]
  gn=np.array(res['models']['EdgeGNN'][str(n)]['rank'])[samples];rr=np.array(res['models']['Residual'][str(n)]['rank'])[samples];dif.append(np.mean(gn==1)-np.mean(rr==1))
 boot[str(n)]=[float(np.quantile(dif,.025)),float(np.quantile(dif,.975))]
audit['bootstrap_gain']=boot
pickle.dump((data,res),open('learning/evaluated.pkl','wb'),protocol=4);json.dump(audit,open('learning/audit.json','w'),indent=2)
print('invariance/gradient',{k:v for k,v in audit.items() if isinstance(v,float)});print('source repaired',verifier);print('group bootstrap',boot)
