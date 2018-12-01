import random
import math
import numpy as np



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
        for j in range(1, M): #Column Major
            if(p != BW[i][j]):
                count += 1
                p =BW[i][j]
        Cols[i] = count
    # 5,6)
    for j in range(M): #find number of changes between 0 and 1, for rows
        p = BW[0][j]
        count = 0
        for i in range(1, N): #Row Major
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
    #order: [bias, Density, Degree/Measure of Symettry, maximum horizontal intersections, average horizntal intersections, maximum vertical intersections, average vertical intersections]

    
class MultiPerceptron:
    def __init__(self):
        self.W = np.array( [ [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )],
                    [random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 ),random.uniform( 0.-1, 0.1 )] ] )
    def pTrain(self, inp, L, O, n ): #where L is the label[0,9], O is the label the neuron guessed, inp is the input, n is eta
        print(O)
        for a in range(7):
            self.W[L][a] = self.W[L][a] + n*inp[a]
        for a in range(7):
            self.W[O][a] = self.W[O][a] - n*inp[a]
    
    #TODO make this function work with numpy
    def pProcess(self, x):
        #tomp = [ sum(  x[b]*self.W[a][b] for b in range( len( self.W[a] ) ) ) for a in range( len( self.W ) ) ]
        #for a in range(len(self.W)):
        #    temp = 
            #for b in range(len(self.W[a])):
            #    temp += x[b]*self.W[a][b]
        #    tomp.append(temp)
        dotProd = []
        for WRow in self.W:
            dotProd += np.dot( x, WRow )
        return np.argmax(dotProd)

    def getWeights(self):
        return self.W



def Project5bitPart2( inputs):
    eta = 0.1
    epochs = 1000
    per = MultiPerceptron()
    for i in range(epochs):
        for casey in inputs:
            inV =  feature( casey[0] )
            Out = per.pProcess( inV )
            if(Out != casey[1]):
                per.pTrain(inV, casey[1], Out, eta)
    return per
