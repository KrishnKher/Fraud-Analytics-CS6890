import pandas as pd
import numpy as np

data = pd.read_csv('cluster.csv')

grouped  = data.groupby('Vertex 1')['Vertex 2'].apply(list) # Convert the data to a
index = np.array(grouped.index)
d = grouped.values

G = dict(zip(index,d))

data['Pairs'] = data[['Vertex 1','Vertex 2']].apply(tuple,axis=1)
phi = dict(zip(data['Pairs'],data['Amount']))


k = 8
m = 7120
MAX_MNV = 200
NOP = 100
Rank = {}

for v in data['Vertex 1']:
    G[v].sort(key=lambda x: phi[(v,x)])
    G[v] = G[v][:k]
for v in data['Vertex 1']:
    Rank[v] = dict(zip(G[v],np.arange(1,len(G[v])))) 
    G[v] = set(G[v])

V = list(data['Vertex 1'])+list(data['Vertex 2'])
V_s = [frozenset({e}) for e in V]
C = dict(zip(V,V_s))
S = set(V_s)
print('Intial Number of Clusters:',len(C))
# print(S)

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
    return mnv
print(MNV(600,707))
cnt = len(S)
while len(S)>m:
    #print(len(S))
    min_mnv = np.inf
    for u in V:
        for v in G.get(u,set()):
            cur_mnv = MNV(u,v)
            if cur_mnv< min_mnv:
                C1,C2 = u,v
                min_mnv = cur_mnv
    if(min_mnv >= MAX_MNV):
        break
    S.remove(C[C1])
    S.remove(C[C2])
    n_C = frozenset(set(C[C1])|set(C[C2]))
    C[C1] = n_C
    C[C2] = n_C
    S.add(n_C)

#print(S)
# cluster = set(frozenset(s) for s in C.values())

# print('Final Number of Clusters:',len(cluster))
