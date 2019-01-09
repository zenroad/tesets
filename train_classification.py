import load_dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from classification_net import classification_net

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_FILES = load_dataset.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = load_dataset.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))
train_file_idxs = np.arange(0, len(TRAIN_FILES))
test_file_idxs = np.arange(0, len(TEST_FILES))
current_data, current_label = load_dataset.loadDataFile(TRAIN_FILES[train_file_idxs[0]])
test_current_data, test_current_label = load_dataset.loadDataFile(TRAIN_FILES[test_file_idxs[0]])
# print TRAIN_FILES[train_file_idxs[0]]
# print current_data.shape, np.max(current_label)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_EPOCHS = 15
BATCH_SIZE = 4
NUM_POINT = 2048

model = classification_net()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

model.to(device)
for epoch in range(MAX_EPOCHS):

    if (epoch%2 != 0):
        for fn in range(len(TRAIN_FILES)):
            current_data, current_label = load_dataset.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
            current_data = current_data[:,0:NUM_POINT,:]
            current_data, current_label, _ = load_dataset.shuffle_data(current_data, np.squeeze(current_label))            
            current_label = np.squeeze(current_label)
            

            file_size = current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            
            total_correct = 0
            total_seen = 0
            loss_sum = 0

            for batch in range(num_batches):
                start_idx = batch * BATCH_SIZE
                end_idx = (batch+1) * BATCH_SIZE
                
                rotated_data = load_dataset.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
                jittered_data = load_dataset.jitter_point_cloud(rotated_data)
                jittered_data = load_dataset.random_scale_point_cloud(jittered_data)
                jittered_data = load_dataset.rotate_perturbation_point_cloud(jittered_data)
                jittered_data = load_dataset.shift_point_cloud(jittered_data)
                
                jittered_data = torch.from_numpy(jittered_data)
                labels = torch.from_numpy(current_label[start_idx:end_idx]).long()
                
                optimizer.zero_grad()
                
                jittered_data = jittered_data.cuda()
                out_labels = model(jittered_data)
                labels = labels.cuda()
                loss = loss_fn(out_labels, labels)
                #print(loss)

                loss.backward()

                optimizer.step()

                #print(epoch)

    if (epoch%2 == 0):
        for fn in range(len(TEST_FILES)):
            test_current_data, test_current_label = load_dataset.loadDataFile(TEST_FILES[test_file_idxs[fn]])
            test_current_data = test_current_data[:,0:NUM_POINT,:]
            test_current_data, test_current_label, _ = load_dataset.shuffle_data(test_current_data, np.squeeze(test_current_label))            
            test_current_label = np.squeeze(test_current_label)
            
            file_size = test_current_data.shape[0]
            num_batches = file_size // BATCH_SIZE
            
            total_correct = 0
            total_seen = 0
            test_loss_sum = 0

            for batch in range(num_batches):
                start_idx = batch * BATCH_SIZE
                end_idx = (batch+1) * BATCH_SIZE
                
                rotated_data = load_dataset.rotate_point_cloud(test_current_data[start_idx:end_idx, :, :])
                jittered_data = load_dataset.jitter_point_cloud(rotated_data)
                jittered_data = load_dataset.random_scale_point_cloud(jittered_data)
                jittered_data = load_dataset.rotate_perturbation_point_cloud(jittered_data)
                jittered_data = load_dataset.shift_point_cloud(jittered_data)
                
                jittered_data = torch.from_numpy(jittered_data)
                labels = torch.from_numpy(test_current_label[start_idx:end_idx]).long()
                
                optimizer.zero_grad()
                
                jittered_data = jittered_data.cuda()
                out_labels = model(jittered_data)
                labels = labels.cuda()
                #loss = loss_fn(out_labels, labels)
                #print(out_labels.size())
                print(labels)
                print(out_labels.data.max(1)[1])
                part_acc = torch.eq(labels,out_labels.data.max(1)[1])
                acc = torch.sum(part_acc)/float(BATCH_SIZE*NUM_POINT)
                print("test acc")
                print(acc)
                