import os,json,pickle,io
os.environ['OMP_NUM_THREADS']='1'
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
r=json.load(open('learning/results.json'));a=json.load(open('learning/inspection.json'));rep=json.load(open('learning/repeats.json'));data,_=pickle.load(open('learning/evaluated.pkl','rb'));dest=Path('figures');dest.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#53677b','axes.labelcolor':'#203c5a','xtick.color':'#425366','ytick.color':'#425366','axes.grid':True,'grid.alpha':.14,'grid.linewidth':.6,'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'dejavuserif','lines.linewidth':1.8})
C={'EdgeGNN':'#007f79','RandomForest':'#c96926','Residual':'#748292','GraphSAGE':'#203c5a','EdgeMLP':'#8a6390','NoPhysics':'#ad8c58'}
L={'EdgeGNN':'Edge-aware GNN','RandomForest':'Random forest','Residual':'Residual ranking','GraphSAGE':'Mean GNN','EdgeMLP':'Endpoint MLP','NoPhysics':'Without residual features'}
def save(fig,name):
 b=io.BytesIO();fig.savefig(b,format='png',dpi=600,bbox_inches='tight',pad_inches=.12);raw=b.getvalue();Image.open(io.BytesIO(raw)).verify();(dest/(name+'.png')).write_bytes(raw);plt.close(fig)
# success and cost
fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),layout='constrained')
for name in ['EdgeGNN','RandomForest','GraphSAGE','EdgeMLP','Residual']:
 y=[r['models'][name][str(n)]['top1']*100 for n in [14,30,57,118]];axs[0].plot(range(4),y,'o-',color=C[name],label=L[name]);axs[1].plot(range(4),[r['models'][name][str(n)]['mean_rank'] for n in [14,30,57,118]],'o-',color=C[name])
for ax in axs:ax.set(xticks=range(4),xticklabels=['14','30','57','118*'],xlabel='Test network (buses)')
axs[0].set(ylabel='Top-1 circuit localization (%)',ylim=(55,96));axs[1].set(ylabel='Mean correct-record rank',yscale='log');axs[0].legend(frameon=False,loc='lower left',fontsize=7.2);save(fig,'Localization-and-Inspection-Rank')
# budget curves
fig,axs=plt.subplots(2,2,figsize=(7.2,5.3),layout='constrained')
for ax,n in zip(axs.ravel(),[14,30,57,118]):
 for name in ['EdgeGNN','RandomForest','GraphSAGE','Residual']:
  ranks=np.array(r['models'][name][str(n)]['rank']);ax.step(range(1,11),[np.mean(ranks<=k)*100 for k in range(1,11)],where='post',color=C[name],label=L[name])
 ax.set(title=f'{n}-bus'+(' transfer' if n==118 else ' withheld circuits'),xlabel='Inspected candidates K',ylabel='Correct circuit in top-K (%)',xticks=[1,3,5,10],ylim=(55,101))
axs[1,0].legend(frameon=False,loc='lower right',fontsize=7.2);save(fig,'Inspection-Budget-Curves')
# precision
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
for name in ['EdgeGNN','RandomForest','Residual']:
 pp=r['breakdown']['precision'];axs[0].plot([2,3,4,5],[pp[str(p)][name]['top1']*100 for p in [2,3,4,5]],'o-',color=C[name],label=L[name]);axs[1].plot([2,3,4,5],[pp[str(p)][name]['mrr'] for p in [2,3,4,5]],'o-',color=C[name])
for ax in axs:ax.set(xticks=[2,3,4,5],xlabel='Printed voltage decimals')
axs[0].set_ylabel('Top-1 circuit localization (%)');axs[1].set_ylabel('Mean reciprocal rank');axs[0].legend(frameon=False,loc='lower right');save(fig,'Reporting-Precision')
# class scores with explicit counts
fig,ax=plt.subplots(figsize=(7.2,3.55),layout='constrained');x=np.arange(6);labels=['Missing','X ×10','X ×0.1','R ×10','Charging\nomitted','Charging\ndoubled']
for j,name in enumerate(['EdgeGNN','RandomForest','Residual']):
 ax.bar(x+(j-1)*.25,[r['breakdown']['kind'][str(k)][name]['top1']*100 for k in range(6)],width=.24,color=C[name],label=L[name])
ax.set(xticks=x,xticklabels=labels,ylabel='Top-1 circuit localization (%)',ylim=(0,118));ax.legend(frameon=False,loc='upper left',ncol=3,fontsize=8)
for k in range(6):ax.text(k,103,'n='+str(r['breakdown']['kind'][str(k)]['N']),ha='center',fontsize=8)
save(fig,'Fault-Family-Results')
# train trends
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),layout='constrained')
for name in ['EdgeGNN','GraphSAGE','EdgeMLP','NoPhysics']:
 z=np.array(r['curves'][name]);axs[0].plot(z[:,0]+1,z[:,1],color=C[name],label=L[name]);axs[1].plot(z[:,0]+1,z[:,2],color=C[name])
axs[0].set(xlabel='Training epoch',ylabel='Mean joint training objective',yscale='log');axs[1].set(xlabel='Training epoch',ylabel='Validation mean reciprocal rank');axs[0].legend(frameon=False,loc='upper right',fontsize=7);save(fig,'Learning-Curves')
# remove residual features
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained');x=np.arange(4)
for j,name in enumerate(['EdgeGNN','NoPhysics']):axs[0].bar(x+(j-.5)*.32,[r['models'][name][str(n)]['top1']*100 for n in [14,30,57,118]],width=.3,color=C[name],label=L[name])
axs[0].set(xticks=x,xticklabels=['14','30','57','118*'],ylabel='Top-1 circuit localization (%)');axs[0].legend(frameon=False,fontsize=7)
for n,col in zip([14,30,57,118],['#007f79','#203c5a','#c96926','#8a6390']):
 rr=a['reliability'][str(n)];axs[1].plot(rr['coverage'],np.array(rr['risk'])*100,color=col,label=str(n)+' bus')
axs[1].set(xlabel='Retained fraction by confidence',ylabel='Localization error (%)',ylim=(0,100));axs[1].legend(frameon=False,loc='upper left',ncol=2,fontsize=7);save(fig,'Physics-and-Confidence')
# reliability
fig,axs=plt.subplots(1,2,figsize=(7.2,3.2),layout='constrained')
for ax,ns,title in [(axs[0],[14,30,57],'Withheld-circuit tests'),(axs[1],[118],'Unseen 118-bus network')]:
 bins=[]
 for i in range(10):
  bb=[z for n in ns for z in a['reliability'][str(n)]['bins'] if abs(z[0]-(i+.5)/10)<1e-6]
  if bb:N=sum(z[1] for z in bb);bins.append([sum(z[1]*z[2] for z in bb)/N,sum(z[1]*z[3] for z in bb)/N,N])
 ax.plot([0,1],[0,1],':',color='gray');ax.scatter([z[0] for z in bins],[z[1] for z in bins],s=[10+z[2]*.25 for z in bins],color=C['EdgeGNN'],alpha=.8);ax.set(title=title,xlabel='Mean predicted confidence',ylabel='Empirical correctness',xlim=(0,1.02),ylim=(0,1.02))
save(fig,'Confidence-Reliability')
# repeated seeds
fig,ax=plt.subplots(figsize=(6.8,3.3),layout='constrained')
for j,seed in enumerate([29,47,83]):
 vals=[(r['models']['EdgeGNN'][str(n)]['top1'] if seed==29 else rep[str(seed)][str(n)]['top1'])*100 for n in [14,30,57,118]];ax.scatter(np.arange(4)+(j-1)*.06,vals,s=30,label='Seed '+str(seed),color=['#007f79','#203c5a','#c96926'][j])
for i,n in enumerate([14,30,57,118]):
 vals=np.array([r['models']['EdgeGNN'][str(n)]['top1'],rep['47'][str(n)]['top1'],rep['83'][str(n)]['top1']])*100;ax.errorbar(i,vals.mean(),yerr=vals.std(ddof=1),fmt='_',color='black',capsize=5)
ax.set(xticks=range(4),xticklabels=['14','30','57','118*'],xlabel='Test network (buses)',ylabel='Top-1 circuit localization (%)',ylim=(65,92));ax.legend(frameon=False,loc='lower left',ncol=3);save(fig,'Training-Repeatability')
# type confusion, explicitly conditional on correct circuit
cm=sum(np.array(a['conditional_type_confusion'][str(n)]) for n in [14,30,57]);mat=cm/cm.sum(1,keepdims=True)*100
fig,ax=plt.subplots(figsize=(6.3,4.55),layout='constrained');im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=100);lab=['Missing','X ×10','X ×0.1','R ×10','Charging 0','Charging ×2'];ax.set(xticks=range(6),yticks=range(6),xticklabels=lab,yticklabels=lab,xlabel='Predicted fault family at true circuit',ylabel='True fault family');ax.tick_params(axis='x',rotation=35);ax.grid(False)
for i in range(6):
 for j in range(6):ax.text(j,i,f'{mat[i,j]:.0f}',ha='center',va='center',color='white' if mat[i,j]>55 else '#203c5a',fontsize=8)
fig.colorbar(im,ax=ax,label='Row percentage');save(fig,'Conditional-Type-Confusion')
# bounds subgroup
fig,ax=plt.subplots(figsize=(6.9,3.4),layout='constrained');subs=[('Row-inconclusive',lambda z:z['lower']<=.001),('Row-excluded',lambda z:z['lower']>.001)]
ss=sum([data[n,'test'] for n in [14,30,57]],[])
for j,name in enumerate(['EdgeGNN','RandomForest','Residual']):
 ys=[]
 for label,rule in subs:
  zz=[z for z in ss if rule(z)];ys.append(np.mean([np.argmax(z['scores'][name])==z['target'] for z in zz])*100)
 ax.bar(np.arange(2)+(j-1)*.24,ys,width=.23,color=C[name],label=L[name])
for i,(label,rule) in enumerate(subs):ax.text(i,105,'n='+str(sum(rule(z) for z in ss)),ha='center',fontsize=8)
ax.set(xticks=[0,1],xticklabels=[z[0] for z in subs],ylabel='Top-1 circuit localization (%)',ylim=(0,118));ax.legend(frameon=False,loc='upper left',ncol=3,fontsize=8);save(fig,'Compatibility-Subgroups')
# clear graph explanation of one selected held-out case, not an aggregate metric
import networkx as nx
ss=data[14,'test'];chosen=[z for z in ss if np.argmax(z['scores']['EdgeGNN'])==z['target'] and int(np.where(np.argsort(-z['scores']['Residual'],kind='stable')==z['target'])[0][0])>=3];z=chosen[0] if chosen else ss[0];edges=z['edges'];present=z['edge'][:,0]>.5;G=nx.Graph();G.add_nodes_from(range(14));G.add_edges_from(edges[present]);pos=nx.spring_layout(G,seed=14,k=.8)
fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained');ax=axs[0]
for i,(u,v) in enumerate(edges):
 xy=np.array([pos[u],pos[v]]);ax.plot(xy[:,0],xy[:,1],color='#c4ccd2' if present[i] else '#e0e4e8',lw=.8,ls='-' if present[i] else '--',zorder=1)
p=z['probabilities'];order=np.argsort(-p)
for j in order[:3]:
 u,v=edges[j];xy=np.array([pos[u],pos[v]]);ax.plot(xy[:,0],xy[:,1],color=C['EdgeGNN'],lw=1.5+2*p[j],alpha=.8,zorder=2)
u,v=edges[z['target']];xy=np.array([pos[u],pos[v]]);ax.plot(xy[:,0],xy[:,1],color=C['RandomForest'],lw=2,ls='--',zorder=3)
for i in range(14):ax.scatter(*pos[i],s=160,color='white',edgecolor='#203c5a',zorder=4);ax.text(*pos[i],str(i+1),ha='center',va='center',fontsize=7,zorder=5)
ax.set_title('Selected withheld-circuit example');ax.axis('off');top=order[:5];axs[1].barh(range(5),p[top],color=C['EdgeGNN']);axs[1].set(yticks=range(5),yticklabels=[f'{edges[j,0]+1}–{edges[j,1]+1}' for j in top],xlabel='Predicted location probability',ylabel='Candidate circuit');axs[1].invert_yaxis();save(fig,'Candidate-Graph-Example')
json.dump({'example_network':14,'target':int(z['target']),'kind':int(z['kind']),'digits':int(z['p']),'residual_rank':int(np.where(np.argsort(-z['scores']['Residual'],kind='stable')==z['target'])[0][0])+1},open('learning/figure-example.json','w'),indent=2)
print('Generated',len(list(dest.glob('*.png'))),'600-dpi evidence graphics')
