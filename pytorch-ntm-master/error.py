import torch
from IPython.display import Image as IPythonImage
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import attr
import argcomplete
from tasks.InterLeaveTask import dataloader
from train import evaluate
from tasks.InterLeaveTask import InterLeaveTaskModelTraining

#Data in the folder are imported
model = InterLeaveTaskModelTraining()
model.net.load_state_dict(torch.load("./InterLeave-task-1000-batch-5000.model"))

#With trained data set dataloader is used to 
_, x, y, seq = next(iter(dataloader(5000, 2, 2, 1, 20))) #(num_batches, batch_size, seq_width, min_len, max_len)
seq = seq.numpy()
y = y.numpy()
#reconstruct the matrix we want
def reconstruct(seq):
    seq_len = seq.shape[0]
    batch_size =  seq.shape[1]
    seq_width = seq.shape[2]
    result = np.zeros((seq_len, batch_size, seq_width))
    A = seq[0:int(seq_len/2),:,:]
    B = seq[int(seq_len/2):,:,:]

    for i in range(int(seq_len/2)):
          result[(2*i),:batch_size,:] = A[i,:,:]#*0.9
          result[(2*i+1),:batch_size,:] = B[i,:,:]
    return result

seq = reconstruct(seq)

a = seq.reshape(-1)
b = y.reshape(-1)
dif=np.abs(a-b)
less_t=np.where(dif<0.1)
acc=less_t[0].shape[0]/a.shape[0]
print('testing error is')
print(acc)