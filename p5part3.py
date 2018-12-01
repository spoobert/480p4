import random
import math
import numpy as np

def nnWarr( i ):
    return -1 if i == 0 else random.uniform( 0.-1, 0.1 )
#img is a numpy 2d array

    
class NNGuess:
    def __init__(self, L, k, n): #where L is the number of labels, k is the number of features, and n is the number of hidden layer neurons
        self.HLayer = [ self.neuron(k) for i in range(n) ]
        self.OLayer = [ self.neuron(n) for i in range(L) ]

#    def copyCons(self, NN):

    class neuron:
        def __init__(self, k): # where k is the number of features
            self.War = [ nnWarr(i) for i in range(k) ] # bias
        def test(self, x, k):
            res = [ x[a]*self.War[a] for a in range( k ) ]
            r = sum(res)
            if(r > 0):
                return 1
            return 0
        
        def train(self, n, x, y, f, k): #n = eta, x = input, y = output from neuron, f = label
            for i in range(k): 
                self.War[i] = self.War[i] - n*(y-f)*x[i]
    
    def process(self, x, pn, k):
        HOut = [ i.test(x, k) for i in self.HLayer ]
        OOut = [ j.test(HOut, pn) for j in self.OLayer ]
        return np.argmax(OOut)

        
    def Correct(self,x,n, pn, k): #back propogation garbage, n is correction coefficient, x is input feature set, pn is the number of hidden layer neurons
        c = 1
        for i in self.OLayer:
            HLf = [(1/(1+(math.e**(-n*j.test(x, k))))) for j in self.HLayer]
            f = 1/(1+(math.e**(-n * i.test(HLf, pn))))
            HOut = [ j.test(x, k) for j in self.HLayer ]
            d = i.test(HOut, pn)
            OD = ((d-f)*f*(1-f))
            for j in range(pn):
                delH = f * (1-f) * (OD * i.War[j])
                for h in self.HLayer[j].War:
                    h = h + c*delH*x[j]
            
            for j in range(pn): #warning: may be 49 #has to come after the previous for loop
                i.War[j] = i.War[j] + c*OD*HOut[j]

                '''
                d = i.test(x)
                f = 1/(1+(float('e')**((-1)*(n)*(i.WXsum(x)))))
                delta = (d-f)*f*(1-f)
                '''

#TODO numpy has some awful combination of functions that would do this lickety split
def feature2(img):
    feat = [-1]
    for i in range(7):
        for j in range(7):
            summ = 0
            for p in range(4):
                for q in range(4):
                    summ += img[4*i + p][4 * j + q]
            summ /= 16
            feat.append(summ)
    return feat

    '''
    for i in range(4, 28, 4):
        for j in range(4, 28, 4):
            summ = 0
            for k in range(i-4, i):
                for h in range(j-4, j):
                    print()
                    summ += img[k][h]
            summ /= 16
            feat.append(summ)
    
    return feat
    '''







def Project5bitPart3( inputs, n):
    eta = 0.1
    epochs = 1000
    N = NNGuess(10, 50, n)
    for i in range(epochs):
        for casey in inputs:
            inV = feature2(casey[0])
            N.Correct(inV,eta)
    return N

