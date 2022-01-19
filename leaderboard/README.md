# Leaderboard


### Download
```
git clone https://github.com/YichaoLu/Traffic4cast2021 oahciy
# wget https://drive.google.com/file/d/1l6ggSXhYZPm7wwspbAboombgE6Y0stLn/view // https://drive.google.com/file/d/1cDZ4mjyhlgP6dODbbbr1S6IMNFsAfFY-/view

git clone https://github.com/SungbinChoi/traffic4cast2021 sungbinchoi
# does not work - https://drive.google.com/file/d/1iXXp-TqphDuixDw7Hx_-vyzaKojcYvhc/view?usp=sharing
#wget "https://drive.google.com/uc?export=download&id=1iXXp-TqphDuixDw7Hx_-vyzaKojcYvhc" -O sungbinchoi/traffic21.zip
# unzip sungbinchoi/traffic21.zip

git clone https://github.com/jbr-ai-labs/traffic4cast-2021 sevakon
# does not work - https://drive.google.com/file/d/1zD0CecX4P3v5ugxaHO2CQW9oX7_D4BCa/view?usp=sharing
#wget "https://drive.google.com/uc?export=download&id=1zD0CecX4P3v5ugxaHO2CQW9oX7_D4BCa" -O sevakon/weights.zip
#unzip sevakon/weights.zip

git clone https://github.com/bojesomo/Traffic4Cast2021-SwinUNet3D bojesomo
# wget https://drive.google.com/file/d/10zM-oiEjRD1rDlDw1bnx06Dl8Z3K3tNQ/view?usp=sharing

git clone https://github.com/NinaWie/NeurIPS2021-traffic4cast nina
wget "https://polybox.ethz.ch/index.php/s/aBvfKzOFkSsSUQv/download?path=%2F&files=ckpt_upp_patch_d100.pt&downloadStartSecret=2jxmjd6z8en" -O nina/ckpt_upp_patch_d100.pt

git clone https://gitlab.com/alchera/alchera-traffic4cast-2021 alchera


git clone https://github.com/LucaHermes/graph-unet-traffic-prediction luca

```

### Update
```
cd alchera; git pull; cd -
cd bojesomo; git pull; cd -
cd jaysantokhi; git pull; cd -
cd luca; git pull; cd -
cd nina; git pull; cd -
cd oahciy; git pull; cd -
cd sevakon; git pull; cd -
cd sungbinchoi; git pull; cd -
```

### Status git clones

```
find .  -maxdepth 1 -mindepth 1 -type d| xargs -n 1 -I{} sh -c 'echo "***{}***"; cd {}; git remote show origin; git status -uall --ignored; git log -1'

***./resuly***
* remote origin
  Fetch URL: https://github.com/resuly/Traffic4Cast-2021
  Push  URL: https://github.com/resuly/Traffic4Cast-2021
  HEAD branch: main
  Remote branch:
    main tracked
  Local branch configured for 'git pull':
    main merges with remote main
  Local ref configured for 'git push':
    main pushes to main (up to date)
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
commit e241964711442edf5512a9e9eb18d5e1eea7a5f6
Author: resuly <resuly@users.noreply.github.com>
Date:   Tue Dec 14 14:28:35 2021 +1100

    Update README.md
***./sevakon***
* remote origin
  Fetch URL: https://github.com/jbr-ai-labs/traffic4cast-2021
  Push  URL: https://github.com/jbr-ai-labs/traffic4cast-2021
  HEAD branch: dev
  Remote branch:
    dev tracked
  Local branch configured for 'git pull':
    dev merges with remote dev
  Local ref configured for 'git push':
    dev pushes to dev (local out of date)
On branch dev
Your branch is up to date with 'origin/dev'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	weights.zip

Ignored files:
  (use "git add -f <file>..." to include in what will be committed)
	__MACOSX/weights/._.DS_Store
	__MACOSX/weights/densenet/._.DS_Store
	__MACOSX/weights/effnetb5/._.DS_Store
	__MACOSX/weights/unet/._.DS_Store
	weights/.DS_Store
	weights/densenet/.DS_Store
	weights/densenet/BERLIN_1008_1430_densenet_unet_mse_best_val_loss_2019=78.4303.pth
	weights/densenet/CHICAGO_1010_1730_densenet_unet_mse_best_val_loss_2019=41.1579.pth
	weights/densenet/MELBOURNE_1009_1619_densenet_unet_mse_best_val_loss_2019=25.7395.pth
	weights/effnetb5/.DS_Store
	weights/effnetb5/BERLIN_1008_1430_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.pth
	weights/effnetb5/CHICAGO_1012_1035_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.pth
	weights/effnetb5/ISTANBUL_1012_2315_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.pth
	weights/effnetb5/MELBOURNE_1010_0058_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.pth
	weights/unet/.DS_Store
	weights/unet/BERLIN_0806_1425_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.pth
	weights/unet/CHICAGO_0805_0038_vanilla_unet_mse_best_val_loss_2019=42.6634.pth
	weights/unet/ISTANBUL_0805_2317_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.pth
	weights/unet/MELBOURNE_0804_1942_vanilla_unet_mse_best_val_loss_2019=26.7588.pth

nothing added to commit but untracked files present (use "git add" to track)
commit 292fd23739a7f0f1ddd3fb6cd7f7e5d99e7879df
Merge: 6288a7a cf18963
Author: Seva Konyakhin <sevakonyakhin@gmail.com>
Date:   Tue Oct 26 15:34:54 2021 +0100

    Merge branch 'dev' of github.com:jbr-ai-labs/traffic4cast-2021 into dev
***./sungbin***
* remote origin
  Fetch URL: https://github.com/SungbinChoi/traffic4cast2021
  Push  URL: https://github.com/SungbinChoi/traffic4cast2021
  HEAD branch: main
  Remote branch:
    main tracked
  Local branch configured for 'git pull':
    main merges with remote main
  Local ref configured for 'git push':
    main pushes to main (local out of date)
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	test/trained_models/t1m1_BERLIN.pth
	test/trained_models/t1m1_CHICAGO.pth
	test/trained_models/t1m1_ISTANBUL.pth
	test/trained_models/t1m1_MELBOURNE.pth
	test/trained_models/t1m2_BERLIN.pth
	test/trained_models/t1m2_CHICAGO.pth
	test/trained_models/t1m2_ISTANBUL.pth
	test/trained_models/t1m2_MELBOURNE.pth
	test/trained_models/t1m3_BERLIN.pth
	test/trained_models/t1m3_CHICAGO.pth
	test/trained_models/t1m3_ISTANBUL.pth
	test/trained_models/t1m3_MELBOURNE.pth
	test/trained_models/t1m4.pth
	test/trained_models/t1m5.pth
	test/trained_models/t1m6.pth
	test/trained_models/t1m7.pth
	test/trained_models/t2m1.pth
	test/trained_models/t2m2.pth
	test/trained_models/t2m3.pth
	test/trained_models/t2m4.pth
	traffic21.zip

nothing added to commit but untracked files present (use "git add" to track)
commit 80cae4c9a3b1253c8f489576ed979511fac0a0a1
Author: Sungbin Choi <sungbin.choi.1@gmail.com>
Date:   Fri Oct 22 04:39:21 2021 +0900

    Update README.md
***./dninja***
* remote origin
  Fetch URL: https://github.com/LucaHermes/graph-unet-traffic-prediction
  Push  URL: https://github.com/LucaHermes/graph-unet-traffic-prediction
  HEAD branch: main
  Remote branch:
    main tracked
  Local branch configured for 'git pull':
    main merges with remote main
  Local ref configured for 'git push':
    main pushes to main (local out of date)
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
commit 6d2a2fc8dccf2ef5e4d01d8c8c19e6876863eaf4
Author: Rocky <30961397+LucaHermes@users.noreply.github.com>
Date:   Wed Nov 3 09:28:20 2021 +0100

    Update README.md
***./ai4ex***
* remote origin
  Fetch URL: https://github.com/bojesomo/Traffic4Cast2021-SwinUNet3D
  Push  URL: https://github.com/bojesomo/Traffic4Cast2021-SwinUNet3D
  HEAD branch: main
  Remote branch:
    main tracked
  Local branch configured for 'git pull':
    main merges with remote main
  Local ref configured for 'git push':
    main pushes to main (up to date)
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	epoch=36-val_loss=51.812557.ckpt

nothing added to commit but untracked files present (use "git add" to track)
commit 00c043e68d38635369a371ef561f95eab12a717a
Author: bojesomo <positivegenius4real@yahoo.com>
Date:   Sun Oct 31 09:30:01 2021 +0400

    Update README.md
***./jaysantokhi***
* remote origin
  Fetch URL: https://gitlab.com/alchera/alchera-traffic4cast-2021
  Push  URL: https://gitlab.com/alchera/alchera-traffic4cast-2021
  HEAD branch: main
  Remote branch:
    main tracked
  Local branch configured for 'git pull':
    main merges with remote main
  Local ref configured for 'git push':
    main pushes to main (up to date)
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
commit 0df1d4edd72293c459512c2eb93566c265e2dbeb
Merge: 0ce0bb0 d456921
Author: Jay Santokhi <jay@alcheratechnologies.com>
Date:   Wed Oct 20 17:22:34 2021 +0000

    Merge branch 'development' into 'main'
    
    consistent
    
    See merge request alchera/alchera-traffic4cast-2021!4
***./oahciy***
* remote origin
  Fetch URL: https://github.com/YichaoLu/Traffic4cast2021
  Push  URL: https://github.com/YichaoLu/Traffic4cast2021
  HEAD branch: main
  Remote branch:
    main tracked
  Local branch configured for 'git pull':
    main merges with remote main
  Local ref configured for 'git push':
    main pushes to main (local out of date)
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	model/core_competition_model_weights(1).zip
	model/extended_competition_model_weights.zip
	model/model_1.bin
	model/model_2.bin
	model/model_3.bin
	model/model_4.bin
	model/model_5.bin
	model/model_6.bin
	model/model_7.bin
	model/v1_epoch_5.bin
	model/v2_epoch_5.bin
	model/v3_epoch_5.bin
	model/v4_epoch_5.bin
	model/v5_epoch_5.bin
	model/v6_epoch_5.bin
	model/v7_epoch_5.bin
	model/v8_epoch_5.bin
	model/v9_epoch_5.bin

nothing added to commit but untracked files present (use "git add" to track)
commit d41d134826c0c460118728570ced0e195ac8a785
Author: Yichao Lu <YichaoLuCharles@gmail.com>
Date:   Sat Oct 30 01:45:18 2021 -0400

    Update README.md
***./nina***
* remote origin
  Fetch URL: https://github.com/NinaWie/NeurIPS2021-traffic4cast
  Push  URL: https://github.com/NinaWie/NeurIPS2021-traffic4cast
  HEAD branch: master
  Remote branches:
    graph-resnet-docstrings tracked
    master                  tracked
    modify_loss             tracked
    specialprize            new (next fetch will store in remotes/origin)
  Local branch configured for 'git pull':
    master merges with remote master
  Local ref configured for 'git push':
    master pushes to master (local out of date)
On branch master
Your branch is up to date with 'origin/master'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	ckpt_upp_patch_d100.pt

nothing added to commit but untracked files present (use "git add" to track)
commit 541ae061f74fc34f73544e7d679ed602b8d74d82
Author: NinaWie <wnina@student.ethz.ch>
Date:   Sun Oct 24 13:04:21 2021 +0200

    test different stitching methods

```

### Get number of parameters from checkpoints

```
export PYTHONPATH=nina
python nina.py

/home/che/miniconda3/envs/t4c/bin/python /home/che/workspaces/neurips2021-traffic4cast/leaderboard/nina.py
2022-01-03 18:59:34.743983: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2022-01-03 18:59:34.743999: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
oahciy core: 9
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v1_epoch_5.bin: <class 'collections.OrderedDict'>  -> 124256816
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v2_epoch_5.bin: <class 'collections.OrderedDict'>  -> 124256816
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v3_epoch_5.bin: <class 'collections.OrderedDict'>  -> 464874160
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v4_epoch_5.bin: <class 'collections.OrderedDict'>  -> 325274160
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v5_epoch_5.bin: <class 'collections.OrderedDict'>  -> 174512816
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v6_epoch_5.bin: <class 'collections.OrderedDict'>  -> 124246448
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v7_epoch_5.bin: <class 'collections.OrderedDict'>  -> 124256816
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v8_epoch_5.bin: <class 'collections.OrderedDict'>  -> 124256816
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/v9_epoch_5.bin: <class 'collections.OrderedDict'>  -> 124256816
oahciy extended: 7
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_1.bin: <class 'collections.OrderedDict'>  -> 2446768
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_2.bin: <class 'collections.OrderedDict'>  -> 2446768
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_3.bin: <class 'collections.OrderedDict'>  -> 2446768
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_4.bin: <class 'collections.OrderedDict'>  -> 2446768
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_5.bin: <class 'collections.OrderedDict'>  -> 2446768
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_6.bin: <class 'collections.OrderedDict'>  -> 2446768
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//oahciy/model/model_7.bin: <class 'collections.OrderedDict'>  -> 2446768
sungbin core: 16
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m1_BERLIN.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m1_ISTANBUL.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m1_MELBOURNE.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m3_BERLIN.pth: <class 'collections.OrderedDict'>  -> 9780528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m3_MELBOURNE.pth: <class 'collections.OrderedDict'>  -> 9780528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m2_MELBOURNE.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m6.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m5.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m2_ISTANBUL.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m3_ISTANBUL.pth: <class 'collections.OrderedDict'>  -> 9780528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m7.pth: <class 'collections.OrderedDict'>  -> 5400944
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m3_CHICAGO.pth: <class 'collections.OrderedDict'>  -> 9780528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m4.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m2_BERLIN.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m1_CHICAGO.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t1m2_CHICAGO.pth: <class 'collections.OrderedDict'>  -> 7188528
sungbin extended: 4
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t2m1.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t2m2.pth: <class 'collections.OrderedDict'>  -> 7188528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t2m4.pth: <class 'collections.OrderedDict'>  -> 9780528
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sungbin/test/trained_models/t2m3.pth: <class 'collections.OrderedDict'>  -> 9780528
sevakon extended: 11
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/densenet/BERLIN_1008_1430_densenet_unet_mse_best_val_loss_2019=78.4303.pth: <class 'collections.OrderedDict'>  -> 32119696
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/densenet/MELBOURNE_1009_1619_densenet_unet_mse_best_val_loss_2019=25.7395.pth: <class 'collections.OrderedDict'>  -> 32119696
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/densenet/CHICAGO_1010_1730_densenet_unet_mse_best_val_loss_2019=41.1579.pth: <class 'collections.OrderedDict'>  -> 32119696
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/effnetb5/MELBOURNE_1010_0058_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.pth: <class 'collections.OrderedDict'>  -> 30306056
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/effnetb5/ISTANBUL_1012_2315_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.pth: <class 'collections.OrderedDict'>  -> 30306056
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/effnetb5/CHICAGO_1012_1035_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.pth: <class 'collections.OrderedDict'>  -> 30306056
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/effnetb5/BERLIN_1008_1430_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.pth: <class 'collections.OrderedDict'>  -> 30306056
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/unet/BERLIN_0806_1425_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.pth: <class 'collections.OrderedDict'>  -> 31111920
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/unet/MELBOURNE_0804_1942_vanilla_unet_mse_best_val_loss_2019=26.7588.pth: <class 'collections.OrderedDict'>  -> 31111920
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/unet/ISTANBUL_0805_2317_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.pth: <class 'collections.OrderedDict'>  -> 31111920
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//sevakon/weights/unet/CHICAGO_0805_0038_vanilla_unet_mse_best_val_loss_2019=42.6634.pth: <class 'collections.OrderedDict'>  -> 31111920
nina both: 1
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//nina/ckpt_upp_patch_d100.pt: <class 'collections.OrderedDict'>  -> 36700848
ai4ex both: 1
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//ai4ex/epoch=36-val_loss=51.812557.ckpt: <class 'collections.OrderedDict'>  -> 141938806
resuly core: 1
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//resuly/models/checkpoints/Resnet3D.pk: <class 'collections.OrderedDict'>  -> 17323030
resuly extended: 1       
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//resuly/models/checkpoints/SparseUNet.pk: <class 'collections.OrderedDict'>  -> 43080
dninja both: 1
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//dninja/ckpts/GraphUNet/GraphUNet_03-10-2021__16-04-37/: <class 'str'>  -> 5844768
jaysantokhi core: 4
2022-01-03 18:59:46.822372: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcuda.so.1
2022-01-03 18:59:46.897579: E tensorflow/stream_executor/cuda/cuda_driver.cc:328] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2022-01-03 18:59:46.897603: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (parsnip): /proc/driver/nvidia/version does not exist
2022-01-03 18:59:46.897847: I tensorflow/core/platform/cpu_feature_guard.cc:142] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./CoreChallenge_Model/FineTune_CHICAGO.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 260310
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./CoreChallenge_Model/FineTune_BERLIN.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 260310
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./CoreChallenge_Model/FineTune_ISTANBUL.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 260310
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./CoreChallenge_Model/FineTune_MELBOURNE.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 260310
jaysantokhi extended: 4
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./ExtendedChallenge_Model/FineTune_CHICAGO.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 85712
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./ExtendedChallenge_Model/FineTune_BERLIN.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 85712
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./ExtendedChallenge_Model/FineTune_ISTANBUL.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 85712
       /home/che/workspaces/neurips2021-traffic4cast/leaderboard//jaysantokhi/./ExtendedChallenge_Model/FineTune_MELBOURNE.hdf5: <class 'tensorflow.python.keras.engine.functional.Functional'>  -> 85712

Process finished with exit code 0

```
