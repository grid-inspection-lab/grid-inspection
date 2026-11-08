# Power-flow equations, analytic Jacobian, Newton solver, reference row-remainder bound and rounding helpers.
import numpy as np, json, itertools, copy
def Ybus(c):
 Y=np.diag(1j*c['sh']);
 for f,t,r,x,b,a in c['br']:
  f,t=int(f)-1,int(t)-1;y=1/(r+1j*x)
  Y[f,f]+=(y+1j*b/2)/a**2;Y[t,t]+=y+1j*b/2;Y[f,t]-=y/a;Y[t,f]-=y/a
 return Y
def indices(c): return np.arange(1,c['n']),np.array([i for i in range(1,c['n']) if i not in c['pv']],int)
def calc(Y,x):
 n=len(Y);v=x[n:]*np.exp(1j*x[:n]);return v*np.conj(Y@v)
def F(c,x,Y=None):
 if Y is None:Y=Ybus(c)
 a,q=indices(c);s=calc(Y,x);return np.r_[c['pg'][a]-c['pd'][a]-s.real[a],-c['qd'][q]-s.imag[q]]
def jac(c,x,Y=None):
 if Y is None:Y=Ybus(c)
 n=c['n'];v=x[n:];d=x[:n,None]-x[None,:n];G=Y.real;H=Y.imag
 C=G*np.cos(d)+H*np.sin(d);S=G*np.sin(d)-H*np.cos(d)
 P,Q=calc(Y,x).real,calc(Y,x).imag
 pt=v[:,None]*v[None,:]*S;qt=-v[:,None]*v[None,:]*C
 pv=v[:,None]*C;qv=v[:,None]*S
 for i in range(n):pt[i,i]=-Q[i]-H[i,i]*v[i]**2;qt[i,i]=P[i]-G[i,i]*v[i]**2;pv[i,i]=P[i]/v[i]+G[i,i]*v[i];qv[i,i]=Q[i]/v[i]-H[i,i]*v[i]
 a,q=indices(c);return -np.block([[pt[a],pv[a]],[qt[q],qv[q]]])
def solve(c,tol=1e-12):
 n=c['n'];a,q=indices(c);ix=np.r_[a,n+q];x=np.r_[np.zeros(n),c['vs']];hist=[]
 for k in range(40):
  f=F(c,x);hist.append(float(max(abs(f))))
  if hist[-1]<tol:return x,hist
  x[ix]+=np.linalg.solve(jac(c,x)[:,ix],-f)
 raise ValueError('no convergence')
def screen(c,x,e):
 n=c['n'];Y=Ybus(c);a,q=indices(c);J=jac(c,x,Y);b1=abs(J)@e;ev=e[n:];et=e[:n];v=x[n:]+ev
 R=[]
 for kind,ii in [('p',a),('q',q)]:
  for i in ii:
   r=abs(Y[i,i].real if kind=='p' else Y[i,i].imag)*ev[i]**2
   for k in range(n):
    if k==i:continue
    ee=et[i]+et[k];r+=abs(Y[i,k])*(.5*v[i]*v[k]*ee**2+ee*(v[k]*ev[i]+v[i]*ev[k])+ev[i]*ev[k])
   R.append(r)
 R=np.array(R);beta=b1+R;f=F(c,x,Y);lo=np.maximum(0,abs(f)-beta);return lo,beta,b1,R,f

def rounded(x,dv,dt):
 n=len(x)//2;return np.r_[np.deg2rad(np.round(np.rad2deg(x[:n]),dt)),np.round(x[n:],dv)],np.r_[np.full(n,np.deg2rad(.5*10.**-dt)),np.full(n,.5*10.**-dv)]

