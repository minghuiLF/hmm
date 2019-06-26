import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
import re


# State_File='dev_set/State_File'
# Symbol_File='dev_set/Symbol_File'
# Query_File='dev_set/Query_File'
State_File='toy_example/State_File'
Symbol_File='toy_example/Symbol_File'
Query_File='toy_example/Query_File'
K=5
with open(State_File) as f:
    N=int(f.readline())
    stateKey={}

    for i in range(N):
        stateKey[i]=f.readline().rstrip()

    # print(stateKey)
    A=np.ones((N,N))
    A[-1]=0
    A[:,-2]=0
    # print(A)
    for line in f:
        f1,f2,f3=line.split()
        f1,f2,f3=int(f1),int(f2),int(f3)
        A[f1,f2]+=f3
    # print(A)
    for i in range(A.shape[0]):
        s=sum(A[i])
        if s!=0:
            A[i]=A[i]/s
    # print(A)

with open(Symbol_File) as f:
    V=int(f.readline())
    symbolKey=defaultdict(lambda:V)

    for i in range(V):
        symbolKey[f.readline().rstrip()]=i

    # print(symbolKey)
    B=np.ones((N,V+1))
    B[-1]=0
    B[-2]=0

    # print(B)
    for line in f:
        f1,f2,f3=line.split()
        f1,f2,f3=int(f1),int(f2),int(f3)
        B[f1,f2]+=f3
    # print(B)
    for i in range(A.shape[0]):
        s=sum(B[i])
        if s!=0:
            B[i]=B[i]/s
    # print(B)

with open(Query_File) as f:
    def extric(s):
        if s=='': return []
        for i in range(len(s)):
            if s[i] in syb:
                left=extric(s[:i])
                right=extric(s[i+1:])
                return (left + [s[i]] +right)
        return [s]
    output=[]
    for line in f:
        query=line.rstrip()
        query=query.split()
        syb=[',','(',')','/','-','&']
        token=[]
        for subq in query:
            token+=extric(subq)
        seq=[s]

        seq=list(map(lambda x:  symbolKey[x] ,token))
        # for i in range(len(token)):
        #     try:
        #         symbolKey[token[i]]
        #     except :
        #         raise
        # print(token)
        # print(seq)
        T=len(seq)
        viterbi=np.zeros((N,T,K))
        backpoint=np.zeros((N,T,K))
        knumber=np.full(T,K)
        backrank=np.zeros((N,T,K))

        viterbi[:,0,:]=np.expand_dims(A[-2,:]*B[:,seq[0]],1).repeat(K,1)
        backpoint[:,0,:]=N-2
        knumber[0]=1
        backrank[:,0,:]=1
        print(viterbi)
        print(backpoint)
        for t in range(1,T):
            #
            # for i in range(N):
            #     tmp=
            #     for k in range(K):
            tmp=[]
            for kn in range(knumber[t-1]):
                # print(np.expand_dims(viterbi[:,t-1,kn],-1)*A*B[:,seq[t]])
                # print('-'*20)
                tmp+=list(np.expand_dims(viterbi[:,t-1,kn],-1)*A*B[:,seq[t]])
            tmp=np.array(tmp)
            knumber[t]=min(sum([1 if i>0 else 0 for i in tmp[:,0]]),K)
            # print(knumber)
            # print('-'*20)
            # print(tmp)
            # print(tmp.shape)
            if tmp.shape[0]<K:
                tmp=np.pad(tmp,((0,K-tmp.shape[0]),(0,0)),'constant',constant_values=-1)
            # print('\n\n11\n')
            # print(np.sort(tmp,0)[::-1])
            # print('\n\n22\n')
            # print(np.max(tmp,0))
            # print('\n\n\n')
            viterbi[:,t]=np.sort(tmp,0)[::-1][:K,:].T
            backpoint[:,t]=(np.argsort(tmp,0)[::-1][:K,:]%N).T
            backrank[:,t]=(np.argsort(tmp,0)[::-1][:K,:]//N).T
            # print('v:\n')
            # print(viterbi)
            # print('b:\n')
            # print(backpoint)
            # print('br:\n')
            # print(backrank)
        # print('lastvalue:\n')
        # print(viterbi)
        # print('\n')
        # print(backpoint)
        # print('\n')
        # print(backrank)
        # print('*********:\n')

        tmp=[]
        for kn in range(knumber[T-1]):
            # print(viterbi[:,T-1,kn]*A[:,-1])
            # print('-'*20)
            tmp+=list(viterbi[:,T-1,kn]*A[:,-1])
        tmp=np.array(tmp)
        # print(tmp)
        # print(tmp.shape)
        # tmp=viterbi[:,T-1]*A[:,-1]
        # print(tmp)
        bestpathrob=np.sort(tmp,0)[::-1][:K]
        bp=np.argsort(tmp,0)[::-1][:K]%N
        br=np.argsort(tmp,0)[::-1][:K]//N

        # print(bestpathrob)
        # print(bp)
        # print(br)

        for k in range(min(sum([1 if i>0 else 0 for i in tmp]),K)):

            path=[np.log(bestpathrob[k]),N-1,bp[k]]
            p=bp[k]
            r=br[k]
            for i in range(1,T+1):
                next_p=int(backpoint[p,-i,r])
                r=int(backrank[p,-i,r])
                p=next_p
                path.append(p)


            output.append(path[::-1])

    print(output)
    for i in output:
        print(i)
