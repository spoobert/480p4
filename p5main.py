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
        #Part 1 Follows
        #TODO define realTest and realTrain for part 1
        Perceptron = 0
        Bepoch = 0
        eta = 0.1
        epochs = 1000
        error = 1
        neuro = neuron()
        for i in range(epochs):
            for case in realTrain:
                inputVector = feature(case[0])
                processed = neuro.Process(inputVector)
                if( processed != case[1] ):
                    neuro.Correct( eta, inputVector, processed, F( case[1] ) )
            cogito = 0
            alucarD = 0
            for case in realTest:
                Dracula += 1
                inputVector = feature(case[0])
                processed = neuro.Process(inputVector)
                if( processed != case[1] ):
                    cogito += 1
            errorRate = cogito/alucarD
            if(errorRate < error):
                error = errorRate
                Perceptron = neuro
                Bepoch = i
        print('least error Perceptron weights: ',Perceptron.Warr)
        print('min error fraction: ', error )
                
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
        #trainedNeuron2 = MultiPerceptron()


        #llows VVVVVV
        realTrain = zipImLab
        #realTrain = np.concatenate(train0,train1,train2,train3,train4,train5,train6,train7,train8,train9)        
        realTest = testImLab
        #realTest = np.concatenate(test0,test1,test2,test3,test4,test5,test6,test7,test8,test9)
        #trainedNeuron2 = Project5bitPart2( realTrain)
        
        Perceptron2 = MultiPerceptron()
        eta = 0.1
        epochs = 1000
        error = 1
        BestMultiPerceptron = 0
        BMij = 0
        for q in range(epochs):
            for p in realTrain:
                inV = feature(q[0])
                Out = Perceptron2.pProcess(inV)
                if(Out != q[1]):
                    Perceptron2.pTrain(inV, q[1], Out, eta)
            Mij = [ [0]*10 ]*10
            for i in realTest:
                inputVector = feature(i[0])
                outty9000 = Perceptron2.pProcess(inputVector)
                Mij[inputVector][outty9000] += 1
            errorRate = 1-(np.dot(np.identity(10), Mij)/np.sum(Mij))
            if(errorRate < error):
                error = errorRate
                BestMultiPerceptron = Perceptron2
                BMij = Mij
                
            
        print('least error Perceptron weights: ',BestMultiPerceptron.W)
        print('min eror fraction: ', error )
        print('confusion matrix: ', BMij)

        
        '''
        for i in range(10):
                trainImgSet.append( read_idx(f'train{i}') )
        for i in range(10):
                testImgSet.append( read_idx(f'train{i}') )
        for i in range(10):
                for trainImg in trainImgSet[i]:
                        trainedNeuron2 = Project5bitPart2( trainImg, i )
        '''
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
        '''
        ##### BEGIN PART 3 ######
        n = [3, 6, 10, 15, 21]
        
        #llows VVVVVV
        eta = 0.1
        epochs = 1000
        Bepoch = Ben = BNN = BNMij = 0
        Bearoar = 1
        for p in n:
            N = NNGuess(10, 50, p)
            for e in range(epochs):
                for casey in realTrain:
                    inV = feature2(casey[0])
                    N.Correct(inV,eta,p)
                NMij = [ [0]*10 ]*10
                for i in realTest:
                    inputVector = feature2(i[0])
                    outty9000 = N.process(inputVector)
                    NMij[inputVector][outty9000] += 1
                errorRate = 1-(np.dot(np.identity(10), NMij)/np.sum(NMij))
                if(errorRate < Bearoar):
                    Bearoar = errorRate
                    Bepoch = e
                    Ben = p
                    BNN = N
                    BNMij = NMij
        print('# Hidden Layer Neurons: ', Ben)
        print('epoch: ', Bepoch)
        print('error rate: ', Bearoar)
        print('the weights of every neuron:')
        print('\tHidden Layer:')
        for i in BNN.HLayer:
            print(i)
        print('\tOutput Layer:')
        for i in BNN.OLayer:
            print(i)

            
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
        #FINAL PART!!!!
        #TODO input image parsing
        s = input('please input the filename of the image file (without extension):\n')
        #img = s.open()
        img = read_idx(s)
        t1 = BestMultiPerceptron.pProcess(feature(img))
        t2 = BNN.process(feature2(img))
        print("Perceptron's Guess: ", t1)
        print("Neural-Network's Guess: ", t2)

if __name__=="__main__":
        main()
