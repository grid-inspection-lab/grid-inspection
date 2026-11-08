exec(open('learning/study.py').read().split("if __name__=='__main__':")[0])
data,_=pickle.load(open('learning/data.pkl','rb'));train=sum([data[n,'train'] for n in [14,30,57]],[])
A=np.concatenate([x['node'] for x in train]);mu=A.mean(0);sd=A.std(0).clip(.05);ee=np.concatenate([x['edge'] for x in train]);em=ee.mean(0);es=ee.std(0).clip(.05);em[0]=0;es[0]=1
for ss in data.values():
 for x in ss:x['node']=(x['node']-mu)/sd;x['edge']=(x['edge']-em)/es
out={}
for seed in [47,83]:
 torch.manual_seed(seed);rng=np.random.default_rng(seed);net=RankGNN(15,7,'edge',True);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=1e-4);best=-1;state=None
 for epoch in range(55):
  net.train()
  for n in rng.permutation([14,30,57]):
   ss=data[int(n),'train'];order=rng.permutation(len(ss))
   for k in range(0,len(ss),32):
    batch=[ss[i] for i in order[k:k+32]];a,b,ii,y=pack(batch);opt.zero_grad();loc,cl=net(a,b,ii);loss=nn.functional.cross_entropy(loc,y)+.3*nn.functional.cross_entropy(cl[torch.arange(len(batch)),y],torch.tensor([z['kind'] for z in batch]));loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step()
  net.eval();ranks=[]
  for n in [14,30,57]:ranks.extend(metrics(evaluate(net,data[n,'val']),data[n,'val'])['rank'])
  score=np.mean(1/np.array(ranks))
  if score>best:best=score;state=copy.deepcopy(net.state_dict())
 net.load_state_dict(state);out[str(seed)]={}
 for n,split in [(14,'test'),(30,'test'),(57,'test'),(118,'transfer')]:out[str(seed)][str(n)]=metrics(evaluate(net,data[n,split]),data[n,split])
 print('seed',seed,{n:round(v['top1'],3) for n,v in out[str(seed)].items()},flush=True)
json.dump(out,open('learning/repeats.json','w'),indent=2)
