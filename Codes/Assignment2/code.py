import pandas as pd
import numpy as np

data = pd.read_csv('cluster.csv')

grouped  = data.groupby('Vertex 1')['Vertex 2'].apply(list)
index = np.array(grouped.index)
d = grouped.values

G = dict(zip(index,d))

data['Pairs'] = data[['Vertex 1','Vertex 2']].apply(tuple,axis=1)
phi = dict(zip(data['Pairs'],data['Amount']))


k = 32
kt = 2

for v in data['Vertex 1']:
    G[v].sort(key=lambda x: phi[(v,x)])
    G[v] = G[v][:k]
for v in data['Vertex 1']:
    G[v] = set(G[v])


V = list(data['Vertex 1'])+list(data['Vertex 2'])
V_s = [{e} for e in V]
C = dict(zip(V,V_s))
print('Intial Number of Clusters:',len(C))
count = 0
for u in V:
    for v in G.get(u,set()):
        if(v not in C[u] and len(G[u] & G.get(v,set()))>=kt and u in G.get(v,set())):
            C[u] = C[u] | C.get(v,set())
            C[v] = C[u]

cluster = set(frozenset(s) for s in C.values())

print('Final Number of Clusters:',len(cluster))
