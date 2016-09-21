#!/usr/bin/env python3

import csv
from collections import OrderedDict
import cPickle
import argparse

import numpy
import theano
from theano import tensor as T

parser = argparse.ArgumentParser(description='Hierarchical regularization of russian election data.')
parser.add_argument('--load', dest='load', help='filename to load', default="")
parser.add_argument('--save', dest='save', help='filename to save', default="gauss.save")
args = parser.parse_args()

N_PARTIES = 14
N_PARTY_OFFSET = 22

class Area:
    def __init__(self):
        self._size = 0
        self._voted = 0
        self._votes = [0]*N_PARTIES
        self._russia = -1
        self._region = -1
        self._oik = -1
        self._tik = -1
        self._uik = -1

    def normalize(self):
        for i in range(N_PARTIES):
            self._votes[i] /= self._voted

areas = []
region2id = {}
id2region = {}

def getRegion(name):
    if not name in region2id:
        region2id[name] = len(region2id)
        id2region[region2id[name]] = name
    return region2id[name]

with open("table_233_level_4.txt") as fIn:
    tsvIn = csv.reader(fIn, delimiter='\t')
    header = None
    totalArea = Area()
    for index, row in enumerate(tsvIn):
        if 0 != index:
            area = Area()
            area._size = float(row[4])
            for i in range(N_PARTIES):
                area._votes[i] = float(row[22 + i])
            area._voted = float(row[10]) + float(row[11])
            area._id = index + 1000000
            area._russia = getRegion("RUSSIA")
            area._region = getRegion(row[0])
            area._oik = getRegion(row[0] + row[1])
            area._tik = getRegion(row[0] + row[1] + row[2])
            area._uik = getRegion(row[0] + row[1] + row[2] + row[3])
            totalArea._voted += area._voted
            for i in range(N_PARTIES):
                totalArea._votes[i] += area._votes[i]
            area.normalize()
            areas.append(area)
        else:
            header = row[:]
            print(header)
            print(header[N_PARTY_OFFSET])
            print(header[10], header[11])

print("Areas = %d" % len(areas))
print("Regions = %d" % len(region2id))
totalArea.normalize()
print("Avg = (%s)" % " ".join(map(str, totalArea._votes)))

loadedParams = None
if len(args.load) != 0:
    with open(args.load, 'rb') as fIn:
        loadedParams = cPickle.load(fIn)

numpyTarget = numpy.random.uniform(-1.0, 1.0, (len(areas), N_PARTIES)).astype(theano.config.floatX)
for i, area in enumerate(areas):
    for j in range(N_PARTIES):
        numpyTarget[i][j] = area._votes[j]
target = theano.shared(name='target', value=numpyTarget)
if not loadedParams:
    numpyEmbeddings = numpy.random.uniform(-0.001, 0.001, (len(region2id), N_PARTIES)).astype(theano.config.floatX)
    for i in range(len(region2id)):
        numpyEmbeddings[i] = totalArea._votes[:]
    for i, area in enumerate(areas):
        numpyEmbeddings[area._uik] = [0]*5
else:
    numpyEmbeddings = loadedParams[0].eval()
embeddings = theano.shared(name='embeddings', value=numpyEmbeddings)
if not loadedParams:
    numpyMixture = numpy.random.uniform(0.1, 0.2, (len(areas), 4)).astype(theano.config.floatX)
    for i in range(len(areas)):
        numpyMixture[i][:] = [1, -0.0001, 0.0001, -0.0001]
else:
    numpyMixture = loadedParams[1].eval()
mixture = theano.shared(name='mixture', value=numpyMixture)
numpyIndices = numpy.zeros( shape=(len(areas), 5), dtype='int32' )
for i, area in enumerate(areas):
    numpyIndices[i] = [area._russia, area._region, area._oik, area._tik, area._uik]
indices = theano.shared(numpyIndices)
areasIndices = T.arange(len(areas))
params = [embeddings, mixture]

loss = theano.scan( lambda i: 100000*(target[i] - mixture[i][0]*embeddings[indices[i][0]] - mixture[i][1]*embeddings[indices[i][1]] - mixture[i][2]*embeddings[indices[i][2]] - mixture[i][3]*embeddings[indices[i][3]] - embeddings[indices[i][4]])**2 +
                                abs(mixture[i][0]) + 3*abs(mixture[i][1]) + 9*abs(mixture[i][2]) + 27*abs(mixture[i][3]), sequences=areasIndices )[0].sum() + 30000*(embeddings**2).sum()
gradients = T.grad(loss, params)
gradientsMean = theano.scan(lambda g, _: g.mean(), sequences=gradients)[0].mean()
lr = T.scalar(name='lr')
lr = 0.00001

# AdaGrad
updates = OrderedDict()
for param, grad in zip(params, gradients):
    value = param.get_value(borrow=True)
    accu = theano.shared(name='accu', value=numpy.zeros(value.shape).astype(theano.config.floatX))
    accuNew = accu + grad**2
    updates[accu] = accuNew
    updates[param] = param - (lr * grad/T.sqrt(accuNew + 1e-6))

train = theano.function(inputs=[], outputs=[loss, gradientsMean], updates=updates)

for it in range(1000):
    trainResults = train()
    print(lr, trainResults[0])
    print("Russia", embeddings[areas[0]._russia].eval())
    print("Mixture", mixture.sum().eval())
    print("Gradients", trainResults[1].mean())
    lr = lr*0.95

    with open(args.save, 'wb') as fOut:
        cPickle.dump(params, fOut, protocol=cPickle.HIGHEST_PROTOCOL)
    with open('embeddings.save', 'w') as fOut:
        evaledEmbeddings = embeddings.eval()
        for i in range(len(region2id)):
            print >>fOut, i, id2region[i], numpy.array(evaledEmbeddings[i])
    with open('mixture.save', 'w') as fOut:
        evaledMixture = mixture.eval()
        for i, area in enumerate(areas):
            print >>fOut, i, id2region[area._region], id2region[area._oik], id2region[area._tik], id2region[area._uik], numpy.array(evaledMixture[i])

