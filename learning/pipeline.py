exec(open('learning/study.py').read().split("if __name__=='__main__':")[0])
from scipy.optimize import linprog

def candidate_bound(c,z,e):
 f,J,R,beta,row=fastscreen(c,z,e);m,d=J.shape
 sol=linprog(np.r_[np.zeros(d),1.],A_ub=np.r_[np.c_[J,-np.ones(m)],np.c_[-J,-np.ones(m)]],b_ub=np.r_[R-f,R+f],bounds=[(-float(v),float(v)) for v in e]+[(0,None)],method='highs')
 if not sol.success:return float(max(row))
 lam=sol.ineqlin.marginals[m:]-sol.ineqlin.marginals[:m];lam/=max(1,sum(abs(lam)));return max(float(max(row)),float(lam@f-abs(J.T@lam)@e-abs(lam)@R),0.)

def inspection_queue(scenario,order,source,budget=None):
 """Inspect one independently supplied source record at a time; do not infer it from the label."""
 n=scenario['n'];M=len(source['br']);ids=scenario['record_ids']
 encoded={int(i):row.copy() for i,row in zip(ids,scenario['branch'])};records=[]
 for count,j in enumerate(order[:len(order) if budget is None else budget],1):
  j=int(j)
  if j>=M:records.append((j,'no source circuit'));continue
  supplied=source['br'][j]
  if j in encoded and np.array_equal(supplied,encoded[j]):records.append((j,'record agrees with source'));continue
  changed=dict(encoded);changed[j]=supplied.copy();c={k:copy.deepcopy(scenario[k]) for k in ['n','pd','qd','pg','vs','pv','sh']};c['br']=np.array([changed[i] for i in sorted(changed)])
  lower=candidate_bound(c,scenario['state'],scenario['radii'])
  try:
   point,_=solve(c);upper=float(max(abs(F(c,point))));inside=np.all(abs(point-scenario['state'])<=scenario['radii']+1e-11)
  except Exception:inside=False;upper=float('inf')
  if inside and upper<=1e-10:return dict(status='compatible point',record=j,queries=count,lower=lower,upper=upper,arithmetic='float64')
  records.append((j,'excluded' if lower>.001+1e-10 else 'unresolved'))
 return dict(status='unresolved',queries=len(order) if budget is None else min(budget,len(order)),arithmetic='float64')

def pipeline_checks():
 data,res=pickle.load(open('learning/evaluated.pkl','rb'));out={};total=0
 for n,split in [(14,'test'),(30,'test'),(57,'test'),(118,'transfer')]:
  src=network(n);ss=data[n,split];checks=[]
  for z in ss:
   order=np.argsort(-z['scores']['EdgeGNN'],kind='stable');outcome=inspection_queue(z,order,src);rank=int(np.where(order==z['target'])[0][0])+1
   assert outcome['status']=='compatible point' and outcome['record']==z['target'] and outcome['queries']==rank
   checks.append(outcome);total+=1
  short=inspection_queue(ss[0],np.argsort(-ss[0]['scores']['EdgeGNN'],kind='stable'),src,budget=0);assert short['status']=='unresolved'
  out[str(n)]={'N':len(checks),'all_correct_source_records':True,'mean_queries':float(np.mean([z['queries'] for z in checks])),'max_corrected_lower':max(z['lower'] for z in checks),'max_point_residual':max(z['upper'] for z in checks),'budget_zero_status':short['status']}
  print(n,out[str(n)],flush=True)
 json.dump(out,open('learning/pipeline.json','w'),indent=2)
if __name__=='__main__':pipeline_checks()
