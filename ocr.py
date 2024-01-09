"""All the library imports."""
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn


"""List of all variables.
Also mentioned and explained reason of some important variables"""

"""This are paths to dataset.training_folder_path,testing_folder_path are for X and rest two are of Y"""
train_labels_path = "written_name_train_v2.csv"
test_labels_path = "written_name_test_v2.csv"
training_folder_path = "train_v2/train/"
testing_folder_path = "test_v2/test/"

"""Batch size for this code must be 1.This batch size represents the number of images processed at once.
But for this code we will only process 1 image at the item because we will be dividing an image into patches and this patches(8) will serve as a batch.
If not used batch size =1 than torch would give error as same gradients will be calculated multiple times. """
BATCH_SIZE = 1

"""This is dictionary and '`' represents space(' ').If you also need to predict numbers than add 0-9 in dictionary"""
num_classes=27
LABELS = {'`':0,"A":1,"B":2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25,'Z':26}

"""Resize image with height 20 pixcels and width 160 pixcels and also grey-scaling image."""
in_channels=1
img_width = 160
img_height =20
"""num_patches is very parameter.It determines the maximum number of characters our model can predict in one image."""
num_patches = 8
"""This denotes height and width of each patch.Also remember 
img_height*img_width*in_channels = num_patches*embedding_dim"""
patch_size=20
embedding_dim=400 # img_width*img_height/(num_patches)

"""num_layers are the number of self_attention layers in decoder and encoder"""
encoder_num_heads =1
encoder_mlp_size=64
encoder_num_layers=4

decoder_num_heads =1
decoder_mlp_size=64
decoder_num_layers=4

epochs = 5

"""Dataloader class storing x and y in form of tuple in training_tuple and testing tuple"""
training_data = []
testing_data = []
training_tuple = []
testing_tuple = []

class myDataLoader():
    TRAIN_CSV = pd.read_csv(train_labels_path)
    TEST_CSV = pd.read_csv(test_labels_path)
    IMG_Width=img_width
    IMG_Height=img_height
    TRAINING_FOLDER = training_folder_path
    TESTING_FOLDER = testing_folder_path

    def make_training_data(self):
        for f in tqdm(os.listdir(self.TRAINING_FOLDER)):
             try:
                path = os.path.join(self.TRAINING_FOLDER, f)  #conplete path of file
                training_data.append(path)
             except Exception as e:
                pass

        for i in range(len(training_data)):
            if self.TRAIN_CSV['IDENTITY'][i] != self.TRAIN_CSV['IDENTITY'][i]:  #true when self.TRAIN_CSV['IDENTITY'][i] is empty
              continue
            training_tuple.append([training_data[i],self.TRAIN_CSV['IDENTITY'][i]])

    def make_testing_data(self):
        for f in tqdm(os.listdir(self.TESTING_FOLDER)):
             try:
                path = os.path.join(self.TESTING_FOLDER, f)  #conplete path of file
                testing_data.append(path)
             except Exception as e:
                pass
        for i in range(len(testing_data)):
            if self.TEST_CSV['IDENTITY'][i] != self.TEST_CSV['IDENTITY'][i]: #true when self.TEST_CSV['IDENTITY'][i] is empty
              continue
            testing_tuple.append([testing_data[i],self.TEST_CSV['IDENTITY'][i]])

""" class PatchEmbedding used to create patches.
Coverts 1*20*160 input  to 1*8*400 output"""
class PatchEmbedding(nn.Module):
  def __init__(self,
               in_channels:int=1,
               patch_size:int=patch_size,
               embedding_dim:int=embedding_dim):
    super().__init__()

    self.patch_size = patch_size

    """Notice that stride and kernel_size both are equal to patch_size.To get intuition you may see what all these parameters are."""
    self.patcher = nn.Conv2d(in_channels=in_channels,
                             out_channels=embedding_dim,
                             kernel_size=patch_size,
                             stride=patch_size,
                             padding=0)

    self.flatten = nn.Flatten(start_dim=1,
                              end_dim=2)

  def forward(self, x):
    # Create assertion to check that inputs are the correct shape
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {self.patch_size}"

    x_patched = self.patcher(torch.tensor(x,dtype = torch.float32))
    x_flattened = self.flatten(x_patched)
    # Make the returned sequence embedding dimensions are in the right order (batch_size, number_of_patches, embedding_dimension)
    return x_flattened.permute(0, 2, 1).type(torch.float32)

"""class to make single decoder layer of DeTR architecture"""
class DecoderLayer(nn.Module):
  def __init__(self,embed_dim:int=embedding_dim,num_heads:int=decoder_num_heads,batch_first=True,dropout=0.1,mlp_size:int=decoder_mlp_size):
    super().__init__()
    """Initializing object queries with all elements equal to 0 and size 1*8*400"""
    self.object_queries = torch.zeros(BATCH_SIZE,8,embed_dim)  # 8 = num_patches
    self.tgt_mask=None

    """self attention block of decoder"""
    self.multiHeadAttention1 = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads = num_heads,
                                                    batch_first=batch_first,
                                                    dropout = dropout)

    self.layerNorm = nn.LayerNorm(normalized_shape=embed_dim)

    """cross attention block of decoder"""
    self.multiHeadAttention2 = nn.MultiheadAttention(embed_dim=embed_dim,
                                                    num_heads = num_heads,
                                                    batch_first=batch_first,
                                                    dropout = dropout)

    self.mlp = nn.Sequential(
        nn.Linear(in_features=embed_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features=mlp_size,
                  out_features=embed_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self,encoder_output,positional_encoding):
    mHA1 = self.multiHeadAttention1(query=self.object_queries+self.object_queries,
                              key=self.object_queries+self.object_queries,
                              value=self.object_queries,
                              need_weights=False)[0]
    ln1 = self.layerNorm(mHA1) + self.object_queries

    mHA2, _ = self.multiHeadAttention2(query=ln1+self.object_queries,
                                    key=positional_encoding+encoder_output,
                                    value=encoder_output,
                                    need_weights=False)
    ln2 = self.layerNorm(mHA2)+ln1

    mlp1 = self.mlp(ln2)
    output = self.layerNorm(mlp1)+ln2
    return output


"""Since we need in total 4 decoder layers I have oduleList to generate 4 layers of decoderLayer class"""
class Decoder(nn.Module):
  def __init__(self,num_layers=decoder_num_layers):
    super().__init__()

    self.layers = nn.ModuleList(
      [
          DecoderLayer(embed_dim=embedding_dim,num_heads=decoder_num_heads,batch_first=True,dropout=0.1,mlp_size=decoder_mlp_size)
          for _ in range(num_layers)
      ]
    )

    """finally using linear layer to have outfeatures equal to number of classes"""
    self.linearLayer=nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)

  def forward(self,encoder_output,positional_encoding):
    for layer in self.layers:
      x = layer(encoder_output,positional_encoding)

    return self.linearLayer(x)
    

class Model(nn.Module):
    def __init__(self,
               img_width:int=img_width,
               img_height:int=img_height,
               in_channels:int=in_channels,
               patch_size:int=patch_size,
               num_transformer_layers:int=encoder_num_layers,
               embedding_dim:int=embedding_dim,
               mlp_size:int=encoder_mlp_size,
               num_heads:int=encoder_num_heads,
               attn_dropout:int=0,
               mlp_dropout:int=0.1,
               embedding_dropout:int=0.1, # Dropout for patch and position embeddings
               num_classes:int=num_classes): # number of classes
        super().__init__()

        self.num_patches = num_patches

        self.patch_embedding = PatchEmbedding(in_channels=1,patch_size=patch_size,embedding_dim=embedding_dim)

        self.positional_embedding = nn.Parameter(torch.randn(1,self.num_patches,embedding_dim))

        self.dropout = nn.Dropout(p=embedding_dropout)

        """used in built transformer encoder of pytorch because its same as that of "Atention is all you need" """
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                         nhead = num_heads,
                                                         dim_feedforward=mlp_size,
                                                         dropout=0.1,
                                                         activation = "gelu",
                                                         batch_first=True,
                                                         norm_first=True),num_layers=num_transformer_layers)

        self.decoder = Decoder()

    def forward(self,input):
        batch_size = input.shape[0]
        x = self.patch_embedding(input)

        x = self.positional_embedding + x

        x = self.dropout(x)

        x = self.transformer_encoder(x)

        x=self.decoder(x,self.patch_embedding(input))

        return x

def acc_fn(y_preds,y_blob_test):
    count = 0
    for i in range(len(y_preds)):
        if y_preds[i] in [y_blob_test[i]-1,y_blob_test[i],y_blob_test[i]+1]:
            count+=1
    return count

"""Initialising list to store accuracy and loss"""
test_acc = []
final_testing_loss = []
final_training_loss = []
train_acc = []

class Train_Test():

    def __init__(self,epochs:int = epochs):
        self.EPOCHS = epochs
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=0.0001)
    def train(self):
        for epoch in range(self.EPOCHS):
            print(f"Epoch: {epoch}\n------")
            train_loss = 0
            correct=0
            for batch, (str_path, y_tuple) in (enumerate(train_dataloader)):

                """making X from file paths"""
                X = []
                for k in range(len(str_path)):
                  img = cv2.imread(os.path.normpath(str_path[k]), cv2.IMREAD_GRAYSCALE)
                  img = cv2.resize(img, (img_width, img_height))
                  X.append(img)

                X = torch.tensor(X)
                model.train()
                y_pred =    model(X.unsqueeze(1))

                """making Y suitable with X as Y should only contain data which is in dictionary named LABELS"""
                y = [[]]
                for i in range(len(y_tuple)):
                    for j in range(min(len(y_pred[i]),len(y_tuple[i]))):
                        if not(y_tuple[i][j] in LABELS):
                            y[i].append(LABELS['`'])
                            continue
                        y[i].append(LABELS[y_tuple[i][j]])
                    for j in range(len(y_pred[i])//2-len(y_tuple[i])//2):
                        y[i].append(LABELS['`'])
                        if len(y[i]) < len(y_pred[i]):
                            y[i].insert(0,LABELS['`'])
                    y.append([])
                y.pop()
                        
                y = torch.tensor(y)
                batch_loss = 0

                """train loop"""
                for i in range(len(y)):
                    loss = self.loss_fn(y_pred[i,:,:]*27,y[i,:])
                    correct += acc_fn(y_pred[i,:,:].argmax(dim=1),y[i,:])
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    train_loss+=loss
                    batch_loss = loss

                if batch % 10000 == 0:
                    # training_loss_list.append(batch_loss)
                    print(f"{epoch} Looked at {batch * len(X)}/{len(train_dataloader.dataset)} samples.")

            train_loss /= len(train_dataloader)
            print(correct/(len(train_dataloader)*8))
            print(train_loss)
            final_training_loss.append(train_loss)
            train_acc.append(correct/(len(train_dataloader)*8))

            # test_loop
            if (epoch+1)%2 == 0:
                test_loss = 0
                correct = 0
                for batch, (str_path, y_tuple) in (enumerate(test_dataloader)):
                    X = []
                    for k in range(len(str_path)):
                        img = cv2.imread(os.path.normpath(str_path[k]), cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, (img_width, img_height))
                        X.append(img)

                    X = torch.tensor(X)
                    model.eval()
                    y_pred = model(X.unsqueeze(1))

                    y = [[]]
                    for i in range(len(y_tuple)):
                        for j in range(min(len(y_pred[i]),len(y_tuple[i]))):
                            if not(y_tuple[i][j] in LABELS):
                                y[i].append(LABELS['`'])
                                continue
                            y[i].append(LABELS[y_tuple[i][j]])
                        for j in range(len(y_pred[i])//2-len(y_tuple[i])//2):
                            y[i].append(LABELS['`'])
                            if len(y[i]) < len(y_pred[i]):
                                y[i].insert(0,LABELS['`'])
                        y.append([])
                    y.pop()
                            
                    y = torch.tensor(y)
                    batch_loss = 0
                    with torch.inference_mode():
                        for i in range(len(y)):
                            loss = self.loss_fn(y_pred[i,:,:]*27,y[i,:])
                            test_loss+=loss
                            batch_loss = loss
                            correct += acc_fn(y_pred[i,:,:].argmax(dim=1),y[i,:])
                    
                    if batch%10000 == 0:
                        print(batch)
                print(correct/(len(test_dataloader)*8),test_loss/len(test_dataloader))
                test_acc.append(correct/(len(test_dataloader)*8))
                final_testing_loss.append(test_loss/len(test_dataloader))

if __name__ == "__main__":
    dataLoader = myDataLoader()
    dataLoader.make_training_data()
    dataLoader.make_testing_data()

    # Turn datasets into iterables (batches)
    train_dataloader = DataLoader(dataset=training_tuple,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    test_dataloader = DataLoader(dataset=testing_tuple,
                                batch_size=BATCH_SIZE,
                                shuffle=False)

    print(train_dataloader, test_dataloader)
    torch.manual_seed(42)
    model =Model()
    print(model)
    train =Train_Test()
    train.train()
    plt.plot([ i for i in range(len(training_loss_list))], np.array(torch.tensor(training_loss_list).numpy()), label="Train loss")
    plt.title("Training loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.plot([ i for i in range(len(test_loss_list))], np.array(torch.tensor(test_loss_list).numpy()), label="Test loss")
    plt.title("test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.plot([ i for i in range(len(test_acc))], np.array(torch.tensor(test_acc).numpy()), label="Test acc")
    plt.title("test acc curves")
    plt.ylabel("acc")
    plt.xlabel("Epochs")
    plt.legend()

    print("final_training_loss",final_training_loss)
    print("final_testing_loss",final_testing_loss)
    print("train_acc",train_acc)
    print("test_acc",test_acc)

    plt.plot([ i for i in range(len(final_training_loss))], np.array(torch.tensor(final_training_loss).numpy()), label="Test acc")
    plt.title("Final loss curves")
    plt.ylabel("loss")
    plt.xlabel("Epochs")
    plt.legend()