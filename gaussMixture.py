#!/usr/bin/env python3

import csv
from collections import OrderedDict
import numpy
import theano
from theano import tensor as T

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

def getRegion(name):
    if not name in region2id:
        region2id[name] = len(region2id)
    return region2id[name]

with open("table_233_level_4.txt") as fIn:
    tsvIn = csv.reader(fIn, delimiter='\t')
    header = None
    someArea = None
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
            area.normalize()
            areas.append(area)
            someArea = area
        else:
            header = row[:]
            print(header)
            print(header[N_PARTY_OFFSET])
            print(header[10], header[11])

print("Areas = %d" % len(areas))
print("Regions = %d" % len(region2id))

numpyTarget = numpy.random.uniform(-1.0, 1.0, (len(areas), N_PARTIES)).astype(theano.config.floatX)
for i, area in enumerate(areas):
    for j in range(N_PARTIES):
        numpyTarget[i][j] = area._votes[j]
target = theano.shared(name='target', value=numpyTarget)
numpyEmbeddings = numpy.random.uniform(-1.0, 1.0, (len(region2id), N_PARTIES)).astype(theano.config.floatX)
for i, area in enumerate(region2id):
    numpyEmbeddings = someArea._votes[:]
embeddings = theano.shared(name='embeddings', value=numpyEmbeddings)
numpyMixture = numpy.random.uniform(0.1, 0.2, (len(areas), 5)).astype(theano.config.floatX)
for i in range(len(areas)):
    numpyMixture[i][:] = [0, 0, 0, 0, 1]
mixture = theano.shared(name='mixture', value=numpyMixture)
numpyIndices = numpy.zeros( shape=(len(areas), 5), dtype='int32' )
for i, area in enumerate(areas):
    numpyIndices[i] = [area._russia, area._region, area._oik, area._tik, area._uik]
indices = theano.shared(numpyIndices)
areasIndices = T.arange(len(areas))
params = [embeddings, mixture]

loss = theano.scan( lambda i: 100000*(target[i] - mixture[i][0]*embeddings[indices[i][0]] - mixture[i][1]*embeddings[indices[i][1]] - mixture[i][2]*embeddings[indices[i][2]] - mixture[i][3]*embeddings[indices[i][3]] - mixture[i][4]*embeddings[indices[i][4]])**2 +
                                abs(mixture[i][1]) + 10*abs(mixture[i][2]) + 100*abs(mixture[i][3]) + 1000*abs(mixture[i][4]), sequences=areasIndices )[0].sum() + 0.001*(embeddings**2).sum()
gradients = T.grad(loss, params)
lr = T.scalar(name='lr')
lr = 0.001
updates = OrderedDict((p, p - lr*g) for p, g in zip(params, gradients))
train = theano.function(inputs=[], outputs=loss, updates=updates)

for it in range(10):
    print(lr, train())
    print(embeddings[areas[0]._russia].eval())
    lr = lr*0.95
