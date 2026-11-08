exec(open('learning/study.py').read().split("if __name__=='__main__':")[0])
from pypower.api import runpf,ppoption
out={};rng=np.random.default_rng(31)
for n in [14,30,57,118]:
 vmerr=0;angerr=0;je=0;re=0
 for _ in range(2):
  d=FNS[n]();factors=rng.uniform(.85,1.15,n);d['bus'][:,2]*=factors;d['bus'][:,3]*=factors;ans,ok=runpf(d,ppoption(VERBOSE=0,OUT_ALL=0,PF_TOL=1e-11,ENFORCE_Q_LIMS=0));assert ok
  slack=int(np.where(d['bus'][:,1]==3)[0][0]);order=[slack]+[i for i in range(n) if i!=slack];c=network(n);c['pd']=d['bus'][order,2]/100;c['qd']=d['bus'][order,3]/100;x,_=solve(c);refV=ans['bus'][order,7];refA=np.deg2rad(ans['bus'][order,8]-ans['bus'][slack,8]);vmerr=max(vmerr,float(max(abs(x[n:]-refV))));angerr=max(angerr,float(max(abs(x[:n]-refA))));J=jac(c,x);hh=1e-5;eye=np.eye(2*n);fd=np.column_stack([(F(c,x+hh*v)-F(c,x-hh*v))/(2*hh) for v in eye]);je=max(je,float(np.max(abs(J-fd))/np.max(abs(J))))
  xp,e=rounded(x,3,2);old=screen(c,xp,e);fast=fastscreen(c,xp,e);re=max(re,float(np.max(abs(old[3]-fast[2]))));assert je<1e-8 and vmerr<1e-8 and angerr<1e-8 and re<1e-10
 out[str(n)]={'voltage_difference':vmerr,'angle_difference_rad':angerr,'jacobian_relative_error':je,'remainder_implementation_difference':re};print(n,out[str(n)],flush=True)
json.dump(out,open('learning/physics-check.json','w'),indent=2)
