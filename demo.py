from __future__ import division, print_function
import argparse, os, time
import torch
import resnet
import transforms # custom transforms for 4D (T, C, H, W) sequences
import read_file2
import torch.nn.functional as F
import cv2
from PIL import Image
import jsonlines
import sys

# tested on torch==1.2.0, python==2.7.6

parser = argparse.ArgumentParser()
parser.add_argument('--workers', default=2, type=int)
parser.add_argument('--epochs', default=200, type=int)
parser.add_argument('--batch_size', default=10*3, type=int)
parser.add_argument('--frame_length', default=50, type=int)
parser.add_argument('--sampling_rate', default=5, type=int)
parser.add_argument('--lr', default=1e-2, type=float) #0.001
parser.add_argument('--print_freq', default=10, type=int)
parser.add_argument('--resnet_checkpoint', default='resnet50_places365.pth.tar', type=str)
parser.add_argument('--clsf_checkpoint', default='checkpoint.pt', type=str)
parser.add_argument('--input_filename', default='/home/siit/friends_video/1x03.mkv', type=str)
parser.add_argument('--output_filename', default='output', type=str)



class_friends = ['none', 'cafe', 'home-livingroom-Monica', 'home-doorway-Monica', 'home-kitchen-Monica', 'home-livingroom-Ross', 'home-none-Ross', 'home-none-Monica', 'restaurant', 'cafe-doorway', 'home-none-none', 'home-kitchen-none', 'hospital', 'museum', 'museum-none-Ross', 'restaurant-none-Monica', 'home-livingroom-Chandler', 'road-none-none', 'office-none-none', 'home-livingroom-none', 'cafe-kitchen-none', 'home-none-Chandler', 'home-kitchen-Chandler', 'home-doorway-Chandler', 'office-none-Chandler', ' ']
print(len(class_friends))
class clsf(torch.nn.Module):
    def __init__(self):
        super(clsf, self).__init__()
        self.lstm_sc = torch.nn.LSTM(input_size=2048, hidden_size=1024, num_layers=2, batch_first=True)
        self.fc2 = torch.nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=1)
        self.fc2_1 = torch.nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1)
        self.fc2_2 = torch.nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)#torch.nn.Linear(1024, 1)
        self.fc3 = torch.nn.Linear(1024, 25)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        self.lstm_sc.flatten_parameters()
        N, T = x.size(0), x.size(1)
        x = self.lstm_sc(x)[0]
        
        # Scene change
        change = self.fc2(x.transpose(1,2))
        change = self.fc2_1(change)
        change = self.fc2_2(change)
        change = torch.squeeze(change,1)
        
       	M, _ = change.max(1)
      	w = change - M.view(-1,1)
      	w = w.exp()
      	w = w.unsqueeze(1).expand(-1,w.size(1),-1)
      	w = w.triu(1) - w.tril()
      	w = w.cumsum(2)
      	w = w - w.diagonal(dim1=1,dim2=2).unsqueeze(2)
      	ww = w.new_empty(w.size())
      	idx = M>=0
      	ww[idx] = w[idx] + M[idx].neg().exp().view(-1,1,1)
      	idx = ~idx
      	ww[idx] = M[idx].exp().view(-1,1,1)*w[idx] + 1
      	ww = (ww+1e-10).pow(-1)
      	ww = ww/ww.sum(1,True)
      	x = ww.transpose(1,2).bmm(x)

        
        x = x.reshape(N*T, -1)
        x = self.fc3(x)
        x = x.reshape(N*T, -1)
        return x

def main():
    torch.backends.cudnn.benchmark = True
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224,224)),
        
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    ### data loader => MJJ

    checkpoint = torch.load(args.resnet_checkpoint)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    fe = resnet.resnet50()
    fe.load_state_dict(state_dict, False)

    model = torch.nn.Sequential(fe, clsf())
    checkpoint = (torch.load(args.clsf_checkpoint))
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['model'].items()}
    model.load_state_dict(state_dict, False)
    model = torch.nn.DataParallel(model).cuda()
    model.eval()
    cap = cv2.VideoCapture(args.input_filename)
    fps=1; n=0; num_frames=0; out_dict=[]; check=0; frame_counter=0
    images = []
    with torch.no_grad():
        while cap.isOpened():
            ret, image = cap.read()
            frame_counter += 1
            if frame_counter >= (cap.get(cv2.CAP_PROP_FPS)/fps):  
                if not ret:
                    check = 10 #End
                else:
                    image = Image.fromarray(image[:, :, ::-1])
                    images.append(image)
                    check += 1 #Append 10 times.

                if check == 10:
                    check = 0
                    data = transform(images).unsqueeze(0).pin_memory().cuda(non_blocking=True)
                    output = model(data) #data : B*T*C*H*W // B : 1 // T : 10 frames.
                    y_friends = F.softmax(output, 0)
                    top5_value_friends, top5_index_friends = y_friends.topk(5)
                    top5_value_friends, top5_label_friends = top5_value_friends.tolist(), [class_friends[i] for i in top5_index_friends[:, 0]]
                    for i in range(len(images)):
                        num_frames += 1
                        out_dict.append({"type": "location", "class": top5_label_friends[i], "seconds": float(num_frames) * 1.0 / float(fps)})
                        
                    images=[]
                    if not ret:
                        break
                frame_counter = 0
            n += 1
            if n%100 == 0:
                print('Processed {}/{} frames'.format(n, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
                sys.stdout.flush()
            if n == 1e10:
                break

    with jsonlines.open(args.output_filename+'.jsonl', mode='w') as writer:
        writer.write_all(out_dict)
    cap.release()
    print('Done')
   
if __name__ == '__main__':
    main()
