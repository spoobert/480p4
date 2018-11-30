import random
import math
import numpy as np

def F( x ):
    return 1 if x == 7 else 0


def feature( img ): #input NxM image(2d aray [N][M]) x, output length 6 feature array
    # 1)
    N,M = 28,28
    Density = img.sum()/(N*M)
    # 2)
    #TODO assumes img is square 
    #tempMeasureOfSym = [ [ img[i][j] ^ img[j][i] for j in range(M)] for i in range(N)]
    verRefImg = np.flip( img, 0 )
    verRefImg = np.flip( verRefImg, 1 )
    measureOfSym = np.bitwise_xor( img, verRefImg )
    measureOfSym = measureOfSym.sum()
    #for row in tempMeasureOfSym:
    #    measureOfSym += sum( row )
    # 3,4,5,6)
    BW = [[0]*M]*N #initialize a new NxM array
    for i,j in zip( range(N), range(M) ): #thresholding operation
        BW[i][j] = int(img[i][j]<=128)
    # 3,4)
    Cols = [0]*N
    # 5,6)
    Rows = [0]*M
    # 3,4)
    for i in range(N): #find number of changes between 0 and 1, for columns
        p = BW[i][0]
        count = 0
        #TODO off by one error?
        for j in range(M): #Column Major
            if(p != BW[i][j]):
                count += 1
                p =BW[i][j]
        Cols[i] = count
    # 5,6)
    for j in range(M): #find number of changes between 0 and 1, for rows
        p = BW[0][j]
        count = 0
        #TODO off by one?
        for i in range(N): #Row Major
            if(p != BW[i][j]):
                count += 1
                p =BW[i][j]
        Rows[j] = count
    # 3)
    maxInterHoriz = max(Cols)
    # 4)
    aveInterHoriz = sum(Cols)/N#len(Cols)
    # 5)
    maxInterVert = max(Rows)
    # 6)
    aveInterVert = sum(Rows)/M#len(Rows)
    return np.array( [-1,Density, measureOfSym, maxInterHoriz, aveInterHoriz, maxInterVert, aveInterVert] )    
    #order: [Density, Degree/Measure of Symettry, maximum horizontal intersections, average horizntal intersections, maximum vertical intersections, average vertical intersections]


class neuron:
    def __init__(self): #correct
        imin = -0.1
        imax = 0.1
        
        #TODO need chance to be negative -.1 to .1 
        self.Warr = np.array( [ random.uniform( 0.-1, 0.1 ), random.uniform( 0.-1, 0.1 ), random.uniform( 0.-1, 0.1 ), random.uniform( 0.-1, 0.1 ), random.uniform( 0.-1, 0.1 ), random.uniform( 0.-1, 0.1 ), random.uniform( 0.-1, 0.1 )] )
    
    def Process(self, x): #correct, x will become the classifiers for any given image
        '''
        res = []
        for a in range(len(x)):
            res.append(x[a]*self.Warr[a])
        r = sum(res)
        '''
        try:
            r = np.dot( x, self.Warr )
        except:
            print(x)
        if(r > 0):
            return 1
        return 0

    def Correct(self, n, x, y, f): #n = eta, x = input, y = output from process
        for i in range(7):
            self.Warr[i] = self.Warr[i] - n*(y-f)*x[i]




def Project5bit(inputs):#to play with
    eta = 0.1
    epochs = 1000
    neuro = neuron()
    for i in range(epochs):
        for case in inputs:
            inputVector = feature(case[0])
            processed = neuro.Process(inputVector)
            if( processed != case[1] ):
                neuro.Correct( eta, inputVector, processed, F( case[1] ) )
    return neuro
