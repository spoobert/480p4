import mnist 
from p5Part1 import *
from p5part2 import *
import time 
import struct
import numpy as np



def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

imgPixels = 784 
mnistIm = read_idx('train-images-idx3-ubyte')
mnistLa = read_idx('train-labels-idx1-ubyte')
trainIm = mnistIm[:200]
trainLa = mnistLa[:200]
testIm = mnistIm[200:250]
testLa = mnistLa[200:250]
zipImLab = zip(trainIm,trainLa)
testImLab = zip(testIm,testLa)
#trainSet = [ (t[0],t[1]) for t in zipImLab ]#if t[1] == 7 or t[1] == 9 ]
#testSet = [ (t[0],t[1]) for t in testImLab ]#if t[1] == 7 or t[1] == 9 ]

#TODO uncomment below to test Part 2
train0 =  [ t[0] for t in zipImLab if t[1] == 0 ] 
train1 =  [ t[0] for t in zipImLab if t[1] == 1 ] 
train2 =  [ t[0] for t in zipImLab if t[1] == 2 ] 
train3 =  [ t[0] for t in zipImLab if t[1] == 3 ] 
train4 =  [ t[0] for t in zipImLab if t[1] == 4 ] 
train5 =  [ t[0] for t in zipImLab if t[1] == 5 ] 
train6 =  [ t[0] for t in zipImLab if t[1] == 6 ] 
train7 =  [ t[0] for t in zipImLab if t[1] == 7 ] 
train8 =  [ t[0] for t in zipImLab if t[1] == 8 ] 
train9 =  [ t[0] for t in zipImLab if t[1] == 9 ] 
trainImgSet = []
trainImgSet.append(train0)
trainImgSet.append(train1)
trainImgSet.append(train2)
trainImgSet.append(train3)
trainImgSet.append(train4)
trainImgSet.append(train5)
trainImgSet.append(train6)
trainImgSet.append(train7)
trainImgSet.append(train8)
trainImgSet.append(train9)
test0 =  [ t[0] for t in testImLab if t[1] == 0 ] 
test1 =  [ t[0] for t in testImLab if t[1] == 1 ] 
test2 =  [ t[0] for t in testImLab if t[1] == 2 ] 
test3 =  [ t[0] for t in testImLab if t[1] == 3 ] 
test4 =  [ t[0] for t in testImLab if t[1] == 4 ] 
test5 =  [ t[0] for t in testImLab if t[1] == 5 ] 
test6 =  [ t[0] for t in testImLab if t[1] == 6 ] 
test7 =  [ t[0] for t in testImLab if t[1] == 7 ] 
test8 =  [ t[0] for t in testImLab if t[1] == 8 ] 
test9 =  [ t[0] for t in testImLab if t[1] == 9 ] 
testImgSet = []
testImgSet.append(test0)
testImgSet.append(test1)
testImgSet.append(test2)
testImgSet.append(test3)
testImgSet.append(test4)
testImgSet.append(test5)
testImgSet.append(test6)
testImgSet.append(test7)
testImgSet.append(test8)
testImgSet.append(test9)
        

def main():
        '''
        set7 = read_idx('train7')
        train780 = int( len(set7) * .8 )
        train7 = set7[:train780]
        test7 = read_idx('test7')
        train9 = read_idx('train9')
        test9 = read_idx('test9')
        '''
        #TODO use proper inputs for part 1 
        '''
        trainedNeuron = Project5bit( trainSet )
        goodCount = 0
        badCount = 0
        bestNeuron = 0
        testError = 1
        for test in testSet:
                res = trainedNeuron.Process(feature(test[0]))
                if res == 1 and test[1] == 7:
                        goodCount += 1
                        continue
                if res == 0 and test[1] == 9:
                        goodCount += 1 
                        continue 
                badCount += 1
                tempTestError = badCount/(goodCount+badCount)
                if( tempTestError < testError ):
                        testError = tempTestError
                        bestNeuron = trainedNeuron
                print(testError)

        print('least error Perceptron weights: ',bestNeuron.Warr)
        print('min eror fraction: ', testError )
        '''

        ###### BEGIN PART 2 #######
        goodCount = 0
        badCount = 0
        bestNeuron = 0
        testError = 1
        Mij = [ [0]*10 ]*10
        #trainImgSet = []
        #testImgSet = []
        trainedNeuron2 = MultiPerceptron()
        '''
        for i in range(10):
                trainImgSet.append( read_idx(f'train{i}') )
        for i in range(10):
                testImgSet.append( read_idx(f'train{i}') )
        for i in range(10):
                for trainImg in trainImgSet[i]:
                        trainedNeuron2 = Project5bitPart2( trainImg, i )
        '''
        for i in range(10):
                trainedNeuron2 = Project5bitPart2( trainImgSet[i], i )


        for i in range(10):
                for testImg in testImgSet[i]:
                        res = trainedNeuron2.pProcess( testImg )
                        if res == i:
                                goodCount += 1
                                continue
                        badCount += 1
                        Mij[i][res] += 1
                #TODO should error be inside the for testImg in testImgSet loop?
                tempTestError = badCount/(goodCount+badCount)
                if( tempTestError < testError ):
                        testError = tempTestError
                        bestNeuron = trainedNeuron2

        print('least error Perceptron weights: ',bestNeuron.W)
        print('min eror fraction: ', testError )
        print('confusion matrix: ', Mij)

        ##### BEGIN PART 3 ######
        '''
        n = [3, 6, 10, 15, 21]
        best = (0, 0, 0)
        for i in n:
                trainedNeuron = Project5bitPart3(trainSet, i)
                goodCount = 0
                badCount = 0
                for test in testSet:
                        res = trainedNeuron.process(feature2(test[0]) )
                        if test[1] == res:
                                goodCount += 1
                                continue
                        badCount += 1
                tempula = badCount/(goodCount+badCount)
                print(tempula)
                if(tempula > best[1]):
                        best = (trainedNeuron, tempula, i)
        print(best)
        '''

if __name__=="__main__":
        main()