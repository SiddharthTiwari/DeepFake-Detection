import torch
from torch.utils.model_zoo import load_url
import matplotlib.pyplot as plt
from scipy.special import expit
from pathlib import Path
import sys
sys.path.append('..')

from architectures.fornet import FeatureExtractor

from blazeface import FaceExtractor, BlazeFace, VideoReader
from architectures import fornet
from isplutils import utils

model_path=Path('weights/binclass/net-Xception_traindb-ff-deepfake_face-scale_size-224_seed-41/bestval.pth')

# get arguments from the model path
face_policy = str(model_path).split('face-')[1].split('_')[0]
patch_size = int(str(model_path).split('size-')[1].split('_')[0])
net_name = str(model_path).split('net-')[1].split('_')[0]
model_name = '_'.join(model_path.with_suffix('').parts[-2:])



device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
face_policy = 'scale'
face_size = 224
frames_per_video = 32

# Load net
net_class = getattr(fornet, net_name)
# load model
print('Loading model...')
state_tmp = torch.load(model_path, map_location='cpu')
if 'net' not in state_tmp.keys():
    state = OrderedDict({'net': OrderedDict()})
    [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
else:
    state = state_tmp
net: FeatureExtractor = net_class().eval().to(device)

incomp_keys = net.load_state_dict(state['net'], strict=True)
print(incomp_keys)
print('Model loaded!')




transf = utils.get_transformer(face_policy, face_size, net.get_normalizer(), train=False)

facedet = BlazeFace().to(device)
facedet.load_weights("blazeface/blazeface.pth")
facedet.load_anchors("blazeface/anchors.npy")
videoreader = VideoReader(verbose=False)
video_read_fn = lambda x: videoreader.read_frames(x, num_frames=frames_per_video)
face_extractor = FaceExtractor(video_read_fn=video_read_fn,facedet=facedet)


vid_real_faces = face_extractor.process_video('samples/490868123550446422495477631417.mp4')
vid_fake_faces = face_extractor.process_video('samples/284649338838012868101332189709.mp4')

## Predict scores for each frame

# For each frame, we consider the face with the highest confidence score found by BlazeFace (= frame['faces'][0])
faces_real_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_real_faces if len(frame['faces'])] )
faces_fake_t = torch.stack( [ transf(image=frame['faces'][0])['image'] for frame in vid_fake_faces if len(frame['faces'])] )

with torch.no_grad():
    faces_real_pred = net(faces_real_t.to(device)).cpu().numpy().flatten()
    faces_fake_pred = net(faces_fake_t.to(device)).cpu().numpy().flatten()
    
    
    
"""
Print average scores.
An average score close to 0 predicts REAL. An average score close to 1 predicts FAKE.
"""
print('Average score for REAL video: {:.4f}'.format(expit(faces_real_pred.mean())))
print('Average score for FAKE face: {:.4f}'.format(expit(faces_fake_pred.mean())))