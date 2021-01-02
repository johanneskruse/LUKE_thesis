
import numpy as np 

'''
Proof of concept: Logits and softmax probability computations
'''

def softmax(logits):
    
    softmax = []
    q = 0
    for i in logits:
        softmax.append(np.exp(i) / sum(np.exp(logits)))
        q += np.exp(i) / sum(np.exp(logits))
    
    return softmax, q

def logits_func(prob): 
    return np.log(prob / (1+prob))

proba = [0.006382342879717207, 0.2564389982593307, 0.04506310418417042, 0.08901875647497302, 0.0920775142662204, 
        0.04941506219220917, 0.16701976595076282, 0.06772062316943511, 0.22686383262318108]
logits = [-5.053441047668457, -1.3600854873657227, -3.098912477493286, -2.4181292057037354, -2.384345531463623, 
        -3.006721019744873, -1.7888641357421875, -2.6915855407714844, -1.4826263189315796]


softmax(logits)

for i in proba: 
    print(logits_func(i))




def logit2prob(logit):
    return np.exp(logit) / (np.exp(logit) + 1)

# x2 samples of > 0.5
logits = [-4.636117458343506, -4.46367883682251, -3.3428733348846436, 2.175978183746338, -4.3089518547058105, 
        -4.214722633361816, -5.114277362823486, 2.573429822921753, -4.526679039001465]




for i in logits: 
    print(logit2prob(i))
