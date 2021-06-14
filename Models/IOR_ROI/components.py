import numpy as np
import torch
import torch.nn as nn
from . import MDN
from .visual_module.visual_features import FeatureFusion, VGG
from .IORROILstm import IORROILstm
from .fixation_duration import FixationDuration
from .utils import Sampler2D, ROIGenerator, OculomotorBias
import os

first_fix_sampler = Sampler2D(np.load('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/first_fix_dist.npy'))
ob = OculomotorBias('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/ob.mat', 12)
roi_gen = ROIGenerator(400, 300, 30)

device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
feature_extractor = VGG(model='vgg19', fine_tune=False).to(device)
fuser = FeatureFusion(3001).to(device)
iorroi_lstm = IORROILstm(512, 512).to(device)
mdn = nn.Sequential(
    nn.Linear(450, 512),
    nn.Tanh(),
    MDN.MDN(512, 2, 10)
).to(device)
fix_duration = FixationDuration(512, 128).to(device)

feature_extractor.load_state_dict(torch.load('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/weights/vgg19.pth', map_location=torch.device('cpu')))
fuser.load_state_dict(torch.load('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/weights/fuser.pth', map_location=torch.device('cpu')))
iorroi_lstm.load_state_dict(torch.load('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/weights/iorroi.pth', map_location=torch.device('cpu')))
mdn.load_state_dict(torch.load('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/weights/mdn.pth',map_location=torch.device('cpu')))
fix_duration.load_state_dict(torch.load('/home/nvinden/school/SequentialSalience/Models/IOR_ROI/data/weights/fix_duration.pth',map_location=torch.device('cpu')))
