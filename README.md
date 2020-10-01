### Prerequisites
- Install [miniconda]
```
$ wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh
$ chmod +x Miniconda3-py37_4.8.2-Linux-x86_64.sh
$ bash ./Miniconda3-py37_4.8.2-Linux-x86_64.sh -b -f -p /usr/local
$ conda init bash
```

- Create the `deepfake_detection` environment with *environment.yml*
```bash
$ conda env create -f environment.yml
$ conda activate deepfake_detection
```


### The whole pipeline
You need to preprocess the datasets in order to index all the samples and extract faces. 

```
$ python index_data.py --source "videos" #Path to DeepFake Video dataset #"/content/gdrive/My Drive/Sentinel_Task/videos"
$ python extract_faces.py --source "/content/gdrive/My Drive/Sentinel_Task/videos" --videodf "data/df_videos.pkl" --facesfolder "faces/" --facesdf "faces_df/" --checkpoint "tmp/checkpoints"

```

Please note that we use only 32 frames per video. You can easily tweak this parameter in [extract_faces.py](extract_faces.py)

### Train
```
python train_binclass.py \
--net Xception \
--traindb ff-deepfake \
--valdb ff-deepfake \
--faces_df_path faces_df_from_video_0_to_video_0.pkl \
--faces_dir faces \
--face scale \
--size 224 \
--batch 64 \
--lr 1e-5 \
--valint 500 \
--patience 10 \
--maxiter 10000 \
--seed 41 \
--attention \
--device 0
```


### Test

```

python test_model.py \
--model_path weights/binclass/net-Xception_traindb-ff-deepfake_face-scale_size-224_seed-41/bestval.pth \
--testsets ff-deepfake \
--faces_df_path faces_df_from_video_0_to_video_0.pkl  \
--faces_dir faces/ \
--device 0

```


#### Pretrained weights
You can find pretrained weights for the trained model. 
Please refer to this [Google Drive link](https://drive.google.com/file/d/1s-2acICKwDUCbNtVpv01awq9OEfdN8r3/view?usp=sharing).

You can find `bestval.pth` which are the best network weights according to the validation set.




## References
- [EfficientNet PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [Xception PyTorch](https://github.com/tstandley/Xception-PyTorch)
- [Video Face Manipulation Detection Through Ensemble of CNNs](https://github.com/polimi-ispl/icpr2020dfdc/)

## Credits

##
[Image and Sound Processing Lab - Politecnico di Milano](http://ispl.deib.polimi.it/)

This  architecture and codebase is based on the Paper Video Face Manipulation Detection Through Ensemble of CNNs, accepted to ICPR2020 and currently available on arXiv.
- Nicolò Bonettini
- Edoardo Daniele Cannas
- Sara Mandelli
- Luca Bondi
- Paolo Bestagini
