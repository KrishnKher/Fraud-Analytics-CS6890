from os import closerange
from numpy.lib.index_tricks import c_
import pandas as pd
import numpy as np

data = pd.read_csv('cluster.csv')

grouped  = data.groupby('Vertex 1')['Vertex 2'].apply(list) # Convert the data to a
index = np.array(grouped.index)
d = grouped.values

G = dict(zip(index,d))

data['Pairs'] = data[['Vertex 1','Vertex 2']].apply(tuple,axis=1)
phi = dict(zip(data['Pairs'],data['Amount']))


k = 10
m = 7000
MAX_MNV = 2
NOP = 1000
Rank = {}

for v in data['Vertex 1']:
    G[v].sort(reverse = True,key=lambda x: phi[(v,x)])
    G[v] = G[v][:k]
for v in data['Vertex 1']:
    Rank[v] = dict(zip(G[v],np.arange(1,len(G[v])))) 
    G[v] = set(G[v])

V = list(data['Vertex 1'])+list(data['Vertex 2'])
V_s = [{e} for e in V]
C = dict(zip(V,V_s))

print('Intial Number of Clusters:',len(C))

def MNV(v1,v2):
    mnv = 0
    for v in C[v1]:
        for u in C[v2]:
            if Rank.get(u,0)!=0:
                mnv += Rank[u].get(v,NOP)
            else:
                mnv += NOP
            if Rank.get(v,0)!=0:
                mnv += Rank[v].get(u,NOP)
            else:
                mnv += NOP
    return mnv/(len(C[v1])*len(C[v2]))

def closeness(v1,v2):
    c_val = 0
    for u in C[v1]:
        for v in C[v2]:
            c_val += phi.get((u,v),0)
            c_val += phi.get((v,u),0)
    return c_val

cnt = len(C)
while cnt>m:
    min_mnv = np.inf
    affinity = 0
    for u in V:
        for v in G.get(u,set()):
            if(u in C[v]):
                break
            cur_mnv = MNV(u,v)
            if cur_mnv< min_mnv:
                C1,C2 = u,v
                min_mnv = cur_mnv
                affinity = closeness(u,v)
            elif cur_mnv==min_mnv:
                cur_aff = closeness(u,v)
                if(affinity<cur_aff):
                    C1,C2 = u,v
                    min_mnv = cur_mnv
                    affinity = closeness(u,v)

    if(min_mnv > MAX_MNV):
        break
    cnt -= 1
    n_C = C[C1]|C[C2]
    for e in C[C1]:
        C[e] = n_C
    for e in C[C2]:
        C[e] = n_C 


cluster = set(tuple(s) for s in C.values())
for c in cluster:
    if(len(c)>1):
        print(c)

print('Final Number of Clusters:',len(cluster))
