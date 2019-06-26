import numpy as np
import collections

def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
	with open(State_File) as state:
		temp_state = []
		data = state.readlines()
	for line in data:
		data = line.split()
		temp_state.append(data)

	with open(Symbol_File) as symbol:
		temp_symbol = []
		data = symbol.readlines()
	for line in data:
		data = line.split()
		temp_symbol.append(data)

	with open(Query_File) as query:
		temp_query = []
		data = query.readlines()
	for line in data:
		if ',' in line or '(' in line or ')' in line or '/' in line or '-' in line or '&' in line:
			line = line.replace(',', ' , ')
			line = line.replace('(', ' ( ')
			line = line.replace(')', ' ) ')
			line = line.replace('/', ' / ')
			line = line.replace('-', ' - ')
			line = line.replace('&', ' & ')
		data = line.split()
		temp_query.append(data)

	state = []
	for i in range(1,int(temp_state[0][0])+1):
		state.append(temp_state[i][0])
	state = dict(enumerate(state))
	state = dict((i,j) for j,i in state.items())

	symbol = []
	for i in range(1,int(temp_symbol[0][0])+1):
		symbol.append(temp_symbol[i][0])
	symbol = dict(enumerate(symbol))
	symbol = dict((i,j) for j,i in symbol.items())

	num_state = len(state)
	num_symbol = len(symbol)

	query = temp_query
	for i in query:
		for j in range(len(i)):
			if i[j] not in symbol:
				i[j] = num_symbol
	for z,v in symbol.items():
		for i in query:
			for j in range(len(i)):
				if z == i[j]:
					i[j] = v
	dict1 = {}
	for i in range(num_state-1):
		dict1.setdefault(i,{})
	for i in range(int(temp_state[0][0])+1,len(temp_state)):
		dict1[int(temp_state[i][0])][int(temp_state[i][1])] = int(temp_state[i][2])

	transition_matrix = np.zeros((int(temp_state[0][0]),int(temp_state[0][0])))
	for i in range(num_state):
		for j in range(num_state):
			if num_state-1 != i:
				total = sum(dict1[i].values())
				if j == num_state-2:
					continue
				if j in dict1[i].keys() and i in dict1.keys():
					transition_matrix[i,j] = (1 + dict1[i][j])/(total+(1 * num_state) - 1)
				else:
					transition_matrix[i,j] = 1 / (total + (1 * num_state) - 1)
		if i == state['BEGIN']:
			pi=transition_matrix[i,:]

	dict2 = {}
	for i in range(num_symbol):
		dict2.setdefault(i,{})
	for i in range(int(temp_symbol[0][0])+1,len(temp_symbol)):
		dict2[int(temp_symbol[i][0])][int(temp_symbol[i][1])] = int(temp_symbol[i][2])

	emission_matrix = np.zeros((int(temp_state[0][0]),int(temp_symbol[0][0])+1))
	for i in range(num_state):
		for j in range(num_symbol+1):
			if num_state - 1 != i and num_state - 2 != i:
				total = sum(dict2[i].values())
				if j in dict2[i].keys() and i in dict2.keys():
					emission_matrix[i,j] = (1 + dict2[i][j])/(total+(1 * num_symbol) + 1)
				else:
					emission_matrix[i,j] = 1 / (total + (1 * num_symbol) + 1)

	# print(query)
	# print(transition_matrix)
	# print(pi)
	# print(emission_matrix)

	#Viterbi top k
	# result = []
	for q in query:
		T = len(q)
		viterbi = np.zeros((k,T+1,num_state))
		backpointer = np.zeros((k,T+1,num_state))
		#Initialisation
		viterbi[:,0,:] =  pi * emission_matrix[:,q[0]]
		backpointer[:,0,:] = num_state -2
		kkk=[k]*(T+1)
		# print(viterbi)
		# print(backpointer)
		# tmp = []
		kkk[1]=1
		for t in range(1,T):
			for s2 in range(num_state):
				tmp = []
				for s1 in range(num_state):
					for m in range((min(k,kkk[t]))):
						tmp.append((viterbi[m,t-1,s1] * transition_matrix[s1,s2] * emission_matrix[s2,q[t]]))

						# if score > viterbi[m,t,s2]:
						# 	viterbi[m,t,s2] = score
						# 	backpointer[m,t,s2] = s1
					tmp = sorted(tmp,reverse = True)
					print('tmm:\n')
					print(tmp)
				for x in range(k):
					viterbi[x,t,s2] = tmp[x]
					backpointer[x,t,s2] = s1
			print(viterbi)
			print(backpointer)
	# 	# print(tmp)
		# print(viterbi)
		# print(backpointer)


k = 2
State_File ='./toy_example/State_File'
Symbol_File='./toy_example/Symbol_File'
Query_File ='./toy_example/Query_File'
#advanced_decoding(State_File, Symbol_File, Query_File)
top_k_viterbi(State_File, Symbol_File, Query_File, k)
#viterbi_algorithm(State_File, Symbol_File, Query_File)
