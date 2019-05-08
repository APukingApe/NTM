"""InterLeave Task NTM model."""
import random

from attr import attrs, attrib, Factory
import torch
from torch import nn
from torch import optim
import numpy as np

from ntm.aio import EncapsulatedNTM
np.random.seed(0)
#In the dataloader, what comment out is the code that used in different tests.

# def dataloader(num_batches,
#                batch_size,
#                seq_width,
#                min_len,
#                max_len):



#       for batch_num in range(num_batches):
        
#         # All batches have the same sequence length
#         seq_len = np.random.randint(min_len, max_len)
#         #seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
#         seq = np.random.rand(seq_len, batch_size, seq_width) / seq_len
        
#         A = seq[0:int(seq_len/2),:,:]
#         B = seq[int(seq_len/2):,:,:]
#         result = np.zeros((seq_len, batch_size, seq_width))

#         for i in range(int(seq_len/2)):
#           result[(2*i),:,:] = A[i,:,:]
#           result[(2*i+1),:,:] = B[i,:,:]
#         #outp = np.copy(seq)
#         #outp = ((outp[:,1:batch_size,:1] - outp[:,:(batch_size-1),:1]) ** 2 + (outp[:,1:batch_size,1:] - outp[:,0:(batch_size-1),1:]) **2) ** 1/2
#         seq = torch.from_numpy(seq)
        
#         # The input includes an additional channel used for the delimiter
#         inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
#         inp[:seq_len, :, :seq_width] = seq
#         inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel
#         #outp = seq.clone()[:seq_len,:,:]
#         #outp = torch.sum(seq, dim = 0)[np.newaxis, :, :]
#         outp = torch.from_numpy(result)
        
        
def dataloader(num_batches,
               batch_size,
               seq_width,
               min_len,
               max_len):



      for batch_num in range(num_batches):
        
        # All batches have the same sequence length
        seq_len = np.random.randint(min_len, max_len)
        #seq = np.random.binomial(1, 0.5, (seq_len, batch_size, seq_width))
        seq = np.random.rand(seq_len, batch_size, seq_width) 
        
        A = seq[0:int(seq_len/2),:,:]
        B = seq[int(seq_len/2):,:,:]
        result = np.zeros((seq_len, batch_size, seq_width))+0.11
        for i in range(int(seq_len/2)):
          result[(2*i),:batch_size,:] = A[i,:,:]*0.9
          result[(2*i+1),:batch_size,:] = B[i,:,:]
        #outp = np.copy(seq)
        #outp = ((outp[:,1:batch_size,:1] - outp[:,:(batch_size-1),:1]) ** 2 + (outp[:,1:batch_size,1:] - outp[:,0:(batch_size-1),1:]) **2) ** 1/2
        seq = torch.from_numpy(seq)
        
        # The input includes an additional channel used for the delimiter
        inp = torch.zeros(seq_len + 1, batch_size, seq_width + 1)
        inp[:seq_len, :, :seq_width] = seq
        inp[seq_len, :, seq_width] = 1.0 # delimiter in our control channel

        outp = torch.from_numpy(result)
        
        

        yield batch_num+1, inp.float(), outp.float(), seq.float()

@attrs
class InterLeaveTaskParams(object):
    name = attrib(default="InterLeave-task")
    controller_size = attrib(default=100, convert=int)
    controller_layers = attrib(default=1,convert=int)
    num_heads = attrib(default=1, convert=int)
    sequence_width = attrib(default=2, convert=int)
    sequence_min_len = attrib(default=1,convert=int)
    sequence_max_len = attrib(default=20, convert=int)
    memory_n = attrib(default=128, convert=int)
    memory_m = attrib(default=20, convert=int)
    num_batches = attrib(default=5000, convert=int)
    batch_size = attrib(default=2, convert=int)
    rmsprop_lr = attrib(default=1e-4, convert=float)
    rmsprop_momentum = attrib(default=0.9, convert=float)
    rmsprop_alpha = attrib(default=0.95, convert=float)


#
# To create a network simply instantiate the `:class:CopyTaskModelTraining`,
# all the components will be wired with the default values.
# In case you'd like to change any of defaults, do the following:
#
# > params = CopyTaskParams(batch_size=4)
# > model = CopyTaskModelTraining(params=params)
#
# Then use `model.net`, `model.optimizer` and `model.criterion` to train the
# network. Call `model.train_batch` for training and `model.evaluate`
# for evaluating.
#
# You may skip this alltogether, and use `:class:CopyTaskNTM` directly.
#

@attrs
class InterLeaveTaskModelTraining(object):
    params = attrib(default=Factory(InterLeaveTaskParams))
    net = attrib()
    dataloader = attrib()
    criterion = attrib()
    optimizer = attrib()

    @net.default
    def default_net(self):
        # We have 1 additional input for the delimiter which is passed on a
        # separate "control" channel
        net = EncapsulatedNTM(self.params.sequence_width + 1, self.params.sequence_width,
                              self.params.controller_size, self.params.controller_layers,
                              self.params.num_heads,
                              self.params.memory_n, self.params.memory_m)
        return net

    @dataloader.default
    def default_dataloader(self):
        return dataloader(self.params.num_batches, self.params.batch_size,
                          self.params.sequence_width,
                          self.params.sequence_min_len, self.params.sequence_max_len)

    @criterion.default
    def default_criterion(self):
        return nn.BCELoss()

    @optimizer.default
    def default_optimizer(self):
        return optim.RMSprop(self.net.parameters(),
                             momentum=self.params.rmsprop_momentum,
                             alpha=self.params.rmsprop_alpha,
                             lr=self.params.rmsprop_lr)
