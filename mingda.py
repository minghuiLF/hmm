# # Import your files here...
import numpy as np
import collections

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
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
	# print(temp_state)
	# print(' ')
	# print(temp_symbol)
	# print(' ')
	# print(temp_query)

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
	# print(num_state)
	# print(num_symbol)
	
	query = temp_query
	#attach unknown
	for i in query:
		for j in range(len(i)):
			if i[j] not in symbol:
				i[j] = 'UNK'
	#ID each entry
	for k,v in symbol.items():
		for i in query:
			for j in range(len(i)):
				if k == i[j]:
					i[j] = v
	#add BEGIN AND END
	for i in range(len(query)):
		query[i] = [num_state-2] + query[i] + [num_state-1]

	# print(state)
	# print(symbol)
	# print('This is query: ',query)
	# print('\n')

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

	# for i in range(len(transition_matrix)):
	# 	for j in range(len(transition_matrix[0])):
	# 		for k in range(int(temp_state[0][0])+1,len(temp_state)):
	# 			if int(temp_state[k][0]) == i and int(temp_state[k][1]) == j:
	# 				transition_matrix[i][j] = (int(temp_state[k][2]) + 1)/(dict1.get(str(i)) + int(temp_state[0][0]) - 1)
	# 			elif i == int(temp_state[0][0])-2 and j == int(temp_state[0][0])-1:
	# 				transition_matrix[i][j] = 1/(dict1.get(str(i)) + int(temp_state[0][0]) - 1)
	# 	if i == int(temp_state[0][0])-2:
	# 		pi = transition_matrix[i,:]

	# print('This is transition_matrix \n',transition_matrix)
	# print('This is pi',pi)

	#find emission matrix
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

	# for i in range(len(emission_matrix)):
	# 	for j in range(len(emission_matrix[0])):
	# 		for k in range(int(temp_symbol[0][0])+1,len(temp_symbol)):
	# 			if j == int(temp_symbol[0][0]) and i < int(temp_symbol[0][0]):
	# 				emission_matrix[i][j] = 1/(dict2.get(str(i)) + int(temp_symbol[0][0]) + 1)
	# 			elif int(temp_symbol[k][0]) == i and int(temp_symbol[k][1]) == j:
	# 				emission_matrix[i][j] = (int(temp_symbol[k][2]) + 1)/(dict2.get(str(i)) + int(temp_symbol[0][0]) + 1)
	
	# print('This is emission_matrix \n',emission_matrix)
	# k = 3
	# result_list = []
	# #Viterbi Algorithm
	# alpha = []
	# for q in query:
	# 	for i in range(num_state):
	# 		for j in range(len(q)+2):

	# 	alpha = []

	# 	alpha = 

##### num_state,num_symbol, state, symbol,transition matrix,emission matrix,
	result_list = []
	for row in query:
		result_matrix = []
		result_initial = []
		s_list = []  # 标记最优路径下的状态

		# initial
		for x in range(num_state-1):
			try:
				p = emission_matrix[x][row[1]] * transition_matrix[row[0]][x]
				result_initial.append(p)
			except IndexError:
				p = emission_matrix[x][num_symbol] * transition_matrix[row[0]][x]
				result_initial.append(p)
		result_matrix.append(result_initial)
		result = max(result_initial)

		#print('Probability Maxtric = ', result_initial, result)

		for x_query in row[2:-1]:       # for query loop
			if x_query == 'UNK':
				x_query = len(symbol)
			result_next = []
			s_s = []
			for i in range(num_state-1):      # for p_i
				p_StoS = []
				s_p = 0
				s = 0  # s for path 标记result_initial最大概率下的状态s
				for x_state in range(num_state-1):          # for max p
					p = transition_matrix[x_state][i]*result_initial[x_state]
					if p > s_p:
						s = x_state
						s_p = p
					p_StoS.append(p)
				s_s.append(s)
				p_m = max(p_StoS)
				p_i = p_m * emission_matrix[i][x_query]
				result_next.append(p_i)

			s_list.append(s_s)
			result_initial = result_next
			result_matrix.append(result_initial)
			#print('Probability Maxtric = ', result_initial)

		# 求出路径s
		s_p = 0
		s = 0                        # s for path 标记result_initial最大概率下的状态s  last one
		for i in range(len(result_initial)):
			if result_initial[i] > s_p:
				s = i
				s_p = result_initial[i]
		state_list = [s]
		for j in range(len(s_list)):
			a = s_list[-j][state_list[0]]
			state_list.insert(0,a)
		
		state_list = [num_state-2] + state_list + [num_state-1]
		# print(result_initial)
		result = max(result_initial)
		# print(result)
		m = 0
		result = result*transition_matrix[s][-1]
		if result != 0 :
			m = np.log(result)
		state_list.append(m)
		result_list.append(state_list)
	#print('========================================')
	# for i in result_list:
	# 	print(i)

	return result_list

# # # Question 2
# # def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
# #     pass # Replace this line with your implementation...


# # # Question 3 + Bonus
# # def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
# #     pass # Replace this line with your implementation...

# State_File ='./dev_set/State_File'
# Symbol_File='./dev_set/Symbol_File'
# Query_File ='./dev_set/Query_File'
# viterbi_algorithm(State_File, Symbol_File, Query_File)

# State_File ='./toy_example/State_File'
# Symbol_File='./toy_example/Symbol_File'
# Query_File ='./toy_example/Query_File'
# viterbi_algorithm(State_File, Symbol_File, Query_File)

###########################################################################################################
# states = ('Healthy','Fever') #STATE
 
# observations = ('normal', 'cold', 'dizzy','cold','normal') #QUERY
 
# start_probability = {'Healthy': 0.6, 'Fever': 0.4} #BEGIN END
 
# transition_probability = {
#    'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
#    'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
#    }
 
# emission_probability = {
#    'Healthy' : {'normal': 0.6, 'cold': 0.4, 'dizzy': 0.1},
#    'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
#    }

# # Helps visualize the steps of Viterbi.
# def print_dptable(V):
#     s = "    " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
#     for y in V[0]:
#         s += "%.5s: " % y
#         s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
#         s += "\n"
#     print(s)
 
# def viterbi(obs, states, start_p, trans_p, emit_p):
#     V = [{}]
#     path = {}
 
#     # Initialize base cases (t == 0)
#     for y in states:
#         V[0][y] = start_p[y] * emit_p[y][obs[0]]
#         path[y] = [y]
#         # print(V)
#         # print(path)
 
#     # Run Viterbi for t > 0
#     for t in range(1, len(obs)):
#         V.append({})
#         newpath = {}
 
#         for y in states:
#             (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
#             V[t][y] = prob
#             newpath[y] = path[state] + [y]
 
#         # Don't need to remember the old paths
#         path = newpath
 
#     print_dptable(V)
#     (prob, state) = max((V[t][y], y) for y in states)
#     return (prob, path[state])

# def example():
#     return viterbi(observations,
#                    states,
#                    start_probability,
#                    transition_probability,
#                    emission_probability)
# print(example())

# viterbi(observations,
#                    states,
#                    start_probability,
#                    transition_probability,
#                    emission_probability)

#####################################################################################################
# import numpy as np

# def viterbi_alg(A_mat, O_mat, observations):
#     # get number of states
#     num_obs = observations.size
#     num_states = A_mat.shape[0]
#     # initialize path costs going into each state, start with 0
#     log_probs = np.zeros(num_states)
#     # initialize arrays to store best paths, 1 row for each ending state
#     paths = np.zeros( (num_states, num_obs+1 ))
#     paths[:, 0] = np.arange(num_states)
#     # start looping
#     for obs_ind, obs_val in enumerate(observations):
#         # for each obs, need to check for best path into each state
#         for state_ind in range(num_states):
#             # given observation, check prob of each path
#             temp_probs = log_probs + \
#                          np.log(O_mat[state_ind, int(obs_val)]) + \
#                          np.log(A_mat[:, state_ind])
#             # check for largest score
#             best_temp_ind = np.argmax(temp_probs)
#             # save the path with a higher prob and score
#             paths[state_ind,:] = paths[best_temp_ind,:]
#             paths[state_ind,(obs_ind+1)] = state_ind
#             log_probs[state_ind] = temp_probs[best_temp_ind]
#     # we now have a best stuff going into each path, find the best score
#     best_path_ind = np.argmax(log_probs)
#     # done, get out.
#     return (best_path_ind, paths, log_probs)
 
# # main script stuff goes here
# if __name__ == '__main__':
#     # the transition matrix
#     A_mat = np.array([[.6, .4], [.2, .8]])
#     # the observation matrix(emission)
#     O_mat = np.array([[.5, .5], [.15, .85]])
#     # sample heads or tails, 0 is heads, 1 is tails
#     num_obs = 4
#     observations1 = np.random.randn( num_obs )
#     observations1[observations1>0] = 1
#     observations1[observations1<=0] = 0
#     # we have what we need, do viterbi
#     best_path_ind, paths, log_probs = viterbi_alg(A_mat, O_mat, observations1)
#     print ("obs1 is " + str(observations1))
#     print(paths)
#     print ("obs1, best path is" + str(paths[best_path_ind,:]))
#     print("the log probabilities are: ",log_probs)
#     print("the best log probability is: ",log_probs[best_path_ind])

    # # change observations to reflect messed up ratio
    # observations2 = np.random.random(num_obs)
    # observations2[observations2>.85] = 0
    # observations2[observations2<=.85] = 1
    # # majority of the time its tails, now what?
    # best_path_ind, paths, log_probs = viterbi_alg(A_mat, O_mat, observations2)
    # print ("obs2 is " + str(observations1))
    # print ("obs2, best path is" + str(paths[best_path_ind,:]))
    # #have it switch partway?
    # best_path_ind, paths, log_probs = viterbi_alg(A_mat, \
    #                                       O_mat, \
    #                                       np.hstack( (observations1, observations2) ) )
    # print ("obs12 is " + str(np.hstack( (observations1, observations2) ) ))
    # print ("obs12, best path is" + str(paths[best_path_ind,:]))
########################################################################################################
	# pi = pi[0:int(temp_state[0][0])-2]
	# A = transition_matrix[0:int(temp_state[0][0])-2,0:int(temp_state[0][0])-2]
	# O = emission_matrix[0:int(temp_state[0][0])-2,0:int(temp_state[0][0])-2]
	# print(pi)
	# print(A)
	# print(O)
	# query = [0, 0, 1, 2]
	# result = []

	# M = len(query)
	# S = pi.shape[0]

	# alpha = np.zeros((M,S))
	# alpha[:,:] = float('-inf')
	# backpointers = np.zeros((M,S),'int')

	# alpha[0,:] = pi * O[:,query[0]]
	# for t in range(1,M):
	# 	for s2 in range(S):
	# 		for s1 in range(S):	
	# 			score = alpha[t-1,s1]*A[s1,s2]*O[s2,query[t]]
	# 			if score > alpha[t,s2]:
	# 				alpha[t,s2] = score
	# 				backpointers[t,s2] = s1

	# ss = []
	# ss.append(np.argmax(alpha[M-1,:]))
	# for i in range(M-1,0,-1):
	# 	ss.append(backpointers[i,ss[-1]])
	# result.append((list(reversed(ss))+np.max(alpha[M-1,:])))

	# print(result)