#!/usr/bin/env python
import time
import mnist_loader
import network_v0 as netw1
import network as netw

tr, va, te = mnist_loader.load_data_wrapper()
mini_batch_size = 10
eta = 10
epochs = 30

t1 = time.clock()
net = netw.Network([784, 30, 10])
net.SGD(tr, epochs, mini_batch_size, eta, test_data=te)
print "origin algorithm time: {0}".format(time.clock()-t1)

t1 = time.clock()
net1 = netw1.Network([784, 30, 10], mini_batch_size)
net1.SGD(tr, epochs, eta, test_data=te)
print "matrix algorithm time: {0}".format(time.clock()-t1)
