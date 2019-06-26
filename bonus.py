# Import your files here...
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
import re
# Question 1
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    # pass # Replace this line with your implementation...
    # State_File='toy_example/State_File'
    # Symbol_File='toy_example/Symbol_File'
    # Query_File='toy_example/Query_File'

    #==========build A
    with open(State_File) as f:
        N=int(f.readline())
        stateKey={}

        for i in range(N):
            stateKey[f.readline().rstrip()]=i

        for ttt in stateKey.items():
            print(ttt)
        # A=np.ones((N,N))
        A=np.full((N,N),0.3)
        A[-1]=0
        A[:,-2]=0
        # A[0:9]
        for line in f:
            f1,f2,f3=line.split()
            f1,f2,f3=int(f1),int(f2),int(f3)
            A[f1,f2]+=f3
        # A[-2,14]=A[-2,8]*2
        # A[-2,8]=
        A[0,19]=0.3
        print(A[:,19])
        for i in range(A.shape[0]):
            s=sum(A[i])
            if s!=0:
                A[i]=A[i]/s

        # print(A)
    #==========build B
    with open(Symbol_File) as f:
        V=int(f.readline())
        symbolKey=defaultdict(lambda:V)

        for i in range(V):
            symbolKey[f.readline().rstrip()]=i

        # print(symbolKey)
        B=np.ones((N,V+1))
        # B=np.full((N,V+1),0.3)
        B[-1]=0
        B[-2]=0
        B[18:24,:]=0
        # B[21,435]=10000
        # B[22,436]=10000
        # print(B)
        for line in f:
            f1,f2,f3=line.split()
            f1,f2,f3=int(f1),int(f2),int(f3)
            B[f1,f2]+=f3
        # B[12,-1]+=1

        # B[14,-1]=1

        # B[21,-1]+=3

        B[9,6472:41020]=0
        B[0,6472:41020]=0.9
        B[1,6472:41020]=0

        for i in range(B.shape[0]):
            s=sum(B[i])
            if s!=0:
                B[i]=B[i]/s
        # print(B)
    #==========handle Query

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

            #==========viterbi
            T=len(seq)
            viterbi=np.zeros((N,T+1))
            backpoint=np.zeros((N,T+1))
            viterbi[:,0]=A[-2,:]*B[:,seq[0]]
            backpoint[:,0]=N-2

            for t in range(1,T):
                tmp=np.expand_dims(viterbi[:,t-1],-1)*A*B[:,seq[t]]
                viterbi[:,t]=np.max(tmp,0)
                backpoint[:,t]=np.argmax(tmp,0)

            tmp=viterbi[:,T-1]*A[:,-1]
            viterbi[:,T]=np.max(tmp,0)
            backpoint[:,T]=np.argmax(tmp,0)
            bp=np.argmax(tmp,0)


            # print(bp)
            # print(viterbi)
            # print(backpoint)

            bestpathrob=np.max(viterbi[:,T])
            path=[N-1,bp]
            # path=[np.log(bestpathrob),N-1,bp]
            for i in range(2,T+2):
                bp=int(backpoint[bp,-i])
                path.append(bp)


            output.append(path[::-1])

        return output






State_File='dev_set/State_File'
Symbol_File='dev_set/Symbol_File'
Query_File='dev_set/Query_File'

Query_Lable='dev_set/Query_Label'

a=advanced_decoding(State_File, Symbol_File, Query_File)



with open(Query_Lable) as f:
    L=list(map(lambda x: list(map(int,str.rstrip(x).split())),f))
MM=[]
for i in range(len(a)):

    SS=np.array(a[i])-np.array(L[i])
    if sum([1 if i!=0 else 0 for i in SS])>1 :
        print(i,SS)
        print(a[i])
        print(L[i],'\n')

        # print(i,sum([1 if i!=0 else 0 for i in SS]))
    MM.append(sum([1 if i!=0 else 0 for i in SS]))
print('Total Number of Incorrect Labels are:\n')
print(sum(MM))
