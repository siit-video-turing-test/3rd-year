import torch.utils.data as data
import numpy as np

from PIL import Image
import os
import os.path

def default_loader(path):
    return Image.open(path).convert('RGB')

def default_flist_reader(flist, frame_length, offset):
  imlist_raw = []
  list_tmp = []
  with open(flist, 'r') as rf:
    for line in rf.readlines():
      impath, imlabel = line.strip().split()
      imlist_raw.append( (impath, int(imlabel)) )
  for i in range(offset, len(imlist_raw)-frame_length, frame_length):
    tmp = imlist_raw[i:i+frame_length]
    list_tmp.append(tmp)

  return list_tmp
 
'''
def default_flist_reader_test(flist, frame_length):
  imlist_raw = []
  list_tmp = []
  imlist = []
  with open(flist, 'r') as rf:
    for line in rf.readlines():
      impath, imlabel = line.strip().split()
      imlist_raw.append( (impath, int(imlabel)) )
                    
  for i in range(0,len(imlist_raw)-frame_length,frame_length):
    tmp = imlist_raw[i:i+frame_length]
    list_tmp.append(tmp)

  if len(imlist_raw)%frame_length>0:
    tmp = imlist_raw[len(imlist_raw)-frame_length:]
    list_tmp.append(tmp)

  return list_tmp
'''
def default_flist_reader_test(flist, frame_length, sampling_rate):
  imlist_raw = []
  list_tmp = []
  imlist = []
  with open(flist, 'r') as rf:
    for line in rf.readlines():
      impath, imlabel = line.strip().split()
      imlist_raw.append( (impath, int(imlabel)) )

  for offset in range(0, frame_length, sampling_rate):    
  # for offset in range(1):          
    for i in range(offset,len(imlist_raw)-frame_length,frame_length):
      tmp = imlist_raw[i:i+frame_length]
      list_tmp.append(tmp)

  if len(imlist_raw)%frame_length>0:
    tmp = imlist_raw[len(imlist_raw)-frame_length:]
    list_tmp.append(tmp)

  return list_tmp




class ImageFilelist(data.Dataset):
  def __init__(self, root, flist, frame_length, sampling_rate=5, transform=None, target_transform=None,
      loader=default_loader):
    self.root   = root
    self.flist = flist
    self.transform = transform
    self.target_transform = target_transform
    self.loader = loader

    self.frame_length = frame_length
    self.sampling_rate = sampling_rate
    self.offset = np.random.randint(self.frame_length)
    if 'train' in flist: self.imlist = default_flist_reader(flist, frame_length, self.offset)       
    # else: self.imlist = default_flist_reader_test(flist, frame_length)
    else: self.imlist = default_flist_reader_test(flist, frame_length, sampling_rate)

  def __getitem__(self, index):
    img = []
    path = []
    tgt = []

    for i in range(0, self.frame_length, self.sampling_rate):
      impath, target = self.imlist[index][i]
      img_tmp = self.loader(os.path.join(self.root,impath))
      img.append(img_tmp)
      tgt.append(target)
      path.append(impath)

    if self.transform is not None:
      img = self.transform(img)
    if self.target_transform is not None:
      tgt = self.target_transform(tgt)
        
    return img, tgt#, path

  def __len__(self):
    return len(self.imlist)

  def set_diff_offset(self):
    if 'train' in self.flist: 
      self.offset = np.random.randint(self.frame_length)
      self.imlist = default_flist_reader(self.flist, self.frame_length, self.offset)    

