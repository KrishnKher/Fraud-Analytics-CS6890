import pandas as pd
import numpy as np

data = pd.read_csv('cluster.csv')
print(len(data))
data = data.iloc[-7194:]


grouped  = data.groupby('Vertex 1')['Vertex 2'].apply(list)
index = np.array(grouped.index)
d = grouped.values

G = dict(zip(index,d))

data['Pairs'] = data[['Vertex 1','Vertex 2']].apply(tuple,axis=1)
phi = dict(zip(data['Pairs'],data['Amount']))


k = 6
kt = 3

for v in data['Vertex 1']:
    G[v].sort(key=lambda x: phi[(v,x)])
    G[v] = G[v][:k]
for v in data['Vertex 1']:
    G[v] = set(G[v])


V = list(data['Vertex 1'])+list(data['Vertex 2'])
V_s = [{e} for e in V]
C = dict(zip(V,V_s))
print(len(C))
count = 0
for u in V:
    for v in G.get(u,set()):
        if(v not in C[u] and len(G[u] & G.get(v,set()))>=kt and u in G.get(v,set())):
            C[u] = C[u] | C.get(v,set())
            C[v] = C[u]
    #print(u,C.get(v,set()))

cluster = list(C.values())
print(cluster)


# v1 = np.array(data['Vertex 1'])
# v2 = np.array(data['Vertex 2'])


# for i in range(len(data)):
#     phi[(data.iloc[i,0],data.iloc[i,1])] = data.iloc[i,2]
#     if data.iloc[i,0] not in G.keys():
#         G[data.iloc[i,0]]=  set()
#     G[data.iloc[i,0]].add(data.iloc[i,1])

print('DONE')