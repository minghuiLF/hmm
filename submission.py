# Import your files here...
import numpy as np
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
import re
# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
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

    #==========build B
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

            bestpathrob=np.max(tmp,0)
            bp=np.argmax(tmp,0)


            # # print(bp)
            # print('v:\n')
            # print(viterbi)
            # print('bp:\n')
            # print(backpoint)


            path=[np.log(bestpathrob),N-1,bp]
            for i in range(2,T+2):
                bp=int(backpoint[bp,-i])
                path.append(bp)


            output.append(path[::-1])

        return output





# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    # State_File='toy_example/State_File'
    # Symbol_File='toy_example/Symbol_File'
    # Query_File='toy_example/Query_File'
    K=k
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
            T=len(seq)
            viterbi=np.zeros((N,T+1,K))
            backpoint=np.zeros((N,T+1,K))
            knumber=np.full(T+1,K)
            backrank=np.zeros((N,T+1,K))

            viterbi[:,0,:]=np.expand_dims(A[-2,:]*B[:,seq[0]],1).repeat(K,1)
            backpoint[:,0,:]=N-2
            knumber[0]=1
            backrank[:,0,:]=1
            # print(viterbi)
            # print(backpoint)
            for t in range(1,T):
                tmp=[]
                for kn in range(knumber[t-1]):

                    tmp+=list(np.expand_dims(viterbi[:,t-1,kn],-1)*A*B[:,seq[t]])
                tmp=np.array(tmp)
                knumber[t]=min(sum([1 if i>0 else 0 for i in tmp[:,0]]),K)

                if tmp.shape[0]<K:
                    tmp=np.pad(tmp,((0,K-tmp.shape[0]),(0,0)),'constant',constant_values=-1)

                viterbi[:,t]=np.sort(tmp,0)[::-1][:K,:].T
                backpoint[:,t]=(np.argsort(tmp,0)[::-1][:K,:]%N).T
                backrank[:,t]=(np.argsort(tmp,0)[::-1][:K,:]//N).T


            tmp=[]
            for kn in range(knumber[T-1]):

                tmp+=list(viterbi[:,T-1,kn]*A[:,-1])
            tmp=np.array(tmp)
            bestpathrob=np.sort(tmp,0)[::-1][:K]
            bp=np.argsort(tmp,0)[::-1][:K]%N
            br=np.argsort(tmp,0)[::-1][:K]//N



            for k in range(min(sum([1 if i>0 else 0 for i in tmp]),K)):

                path=[np.log(bestpathrob[k]),N-1,bp[k]]
                p=bp[k]
                r=br[k]
                for i in range(2,T+2):
                    next_p=int(backpoint[p,-i,r])
                    r=int(backrank[p,-i,r])
                    p=next_p
                    path.append(p)


                output.append(path[::-1])

        return output



# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    with open(State_File) as f:
        N=int(f.readline())
        stateKey={}

        for i in range(N):
            stateKey[f.readline().rstrip()]=i

        # print(stateKey)
        A=np.ones((N,N))
        A[-1]=0
        A[:,-2]=0

        # print(A)
        for line in f:
            f1,f2,f3=line.split()
            f1,f2,f3=int(f1),int(f2),int(f3)
            A[f1,f2]+=f3*3
        A[0,19]=1
        # print(A)
        for i in range(A.shape[0]):
            s=sum(A[i])
            if s!=0:
                A[i]=A[i]/s

    #==========build B
    with open(Symbol_File) as f:
        V=int(f.readline())
        symbolKey=defaultdict(lambda:V)

        for i in range(V):
            symbolKey[f.readline().rstrip()]=i

        # print(symbolKey)
        B=np.ones((N,V+1))
        B[-1]=0
        B[-2]=0
        B[18:24,:]=0
        B[21,435]=10000
        B[22,436]=10000
        # print(B)
        for line in f:
            f1,f2,f3=line.split()
            f1,f2,f3=int(f1),int(f2),int(f3)
            B[f1,f2]+=f3


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

            bestpathrob=np.max(tmp,0)
            bp=np.argmax(tmp,0)


            # # print(bp)
            # print('v:\n')
            # print(viterbi)
            # print('bp:\n')
            # print(backpoint)


            path=[np.log(bestpathrob),N-1,bp]
            for i in range(2,T+2):
                bp=int(backpoint[bp,-i])
                path.append(bp)


            output.append(path[::-1])

        return output


# a=top_k_viterbi(1, 2, 3,4)
# print(a)
