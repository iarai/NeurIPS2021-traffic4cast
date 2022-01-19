import numpy as np
import tensorflow as tf
import torch

BASE_PATH = "/home/che/workspaces/neurips2021-traffic4cast/leaderboard/"
config = {
    'oahciy': {
        'core': [
            'model/v1_epoch_5.bin',
            'model/v2_epoch_5.bin',
            'model/v3_epoch_5.bin',
            'model/v4_epoch_5.bin',
            'model/v5_epoch_5.bin',
            'model/v6_epoch_5.bin',
            'model/v7_epoch_5.bin',
            'model/v8_epoch_5.bin',
            'model/v9_epoch_5.bin'
        ],
        'extended': ['model/model_1.bin',
                     'model/model_2.bin',
                     'model/model_3.bin',
                     'model/model_4.bin',
                     'model/model_5.bin',
                     'model/model_6.bin',
                     'model/model_7.bin',
                     ]
    },
    'sungbin': {
        'core': [
            'test/trained_models/t1m1_BERLIN.pth',
            'test/trained_models/t1m1_ISTANBUL.pth',
            'test/trained_models/t1m1_MELBOURNE.pth',
            'test/trained_models/t1m3_BERLIN.pth',
            'test/trained_models/t1m3_MELBOURNE.pth',
            'test/trained_models/t1m2_MELBOURNE.pth',
            'test/trained_models/t1m6.pth',
            'test/trained_models/t1m5.pth',
            'test/trained_models/t1m2_ISTANBUL.pth',
            'test/trained_models/t1m3_ISTANBUL.pth',
            'test/trained_models/t1m7.pth',
            'test/trained_models/t1m3_CHICAGO.pth',
            'test/trained_models/t1m4.pth',
            'test/trained_models/t1m2_BERLIN.pth',
            'test/trained_models/t1m1_CHICAGO.pth',
            'test/trained_models/t1m2_CHICAGO.pth', ],
        'extended': [
            'test/trained_models/t2m1.pth',
            'test/trained_models/t2m2.pth',
            'test/trained_models/t2m4.pth',
            'test/trained_models/t2m3.pth',

        ]
    },
    'sevakon': {
        'extended': [
            'weights/densenet/BERLIN_1008_1430_densenet_unet_mse_best_val_loss_2019=78.4303.pth',
            'weights/densenet/MELBOURNE_1009_1619_densenet_unet_mse_best_val_loss_2019=25.7395.pth',
            'weights/densenet/CHICAGO_1010_1730_densenet_unet_mse_best_val_loss_2019=41.1579.pth',
            'weights/effnetb5/MELBOURNE_1010_0058_efficientnetb5_unet_mse_best_val_loss_2019=26.0132.pth',
            'weights/effnetb5/ISTANBUL_1012_2315_efficientnetb5_unet_mse_best_val_loss_2019=55.7918.pth',
            'weights/effnetb5/CHICAGO_1012_1035_efficientnetb5_unet_mse_best_val_loss_2019=41.6425.pth',
            'weights/effnetb5/BERLIN_1008_1430_efficientnetb5_unet_mse_best_val_loss_2019=80.3510.pth',
            'weights/unet/BERLIN_0806_1425_vanilla_unet_mse_best_val_loss_2019=0.0000_v5.pth',
            'weights/unet/MELBOURNE_0804_1942_vanilla_unet_mse_best_val_loss_2019=26.7588.pth',
            'weights/unet/ISTANBUL_0805_2317_vanilla_unet_mse_best_val_loss_2019=0.0000_v4.pth',
            'weights/unet/CHICAGO_0805_0038_vanilla_unet_mse_best_val_loss_2019=42.6634.pth',
        ]
    },
    'nina': {'both': ['ckpt_upp_patch_d100.pt']},
    'ai4ex': {'both': ['epoch=36-val_loss=51.812557.ckpt']},
    'resuly': {
        'core': ['models/checkpoints/Resnet3D.pk'],
        'extended':[         'models/checkpoints/SparseUNet.pk']
    },
    'dninja': {
        'both': ['ckpts/GraphUNet/GraphUNet_03-10-2021__16-04-37/']
    },
    'jaysantokhi': {
        'core': [
            './CoreChallenge_Model/FineTune_CHICAGO.hdf5',
            './CoreChallenge_Model/FineTune_BERLIN.hdf5',
            './CoreChallenge_Model/FineTune_ISTANBUL.hdf5',
            './CoreChallenge_Model/FineTune_MELBOURNE.hdf5',
        ],
        'extended': [
            './ExtendedChallenge_Model/FineTune_CHICAGO.hdf5',
            './ExtendedChallenge_Model/FineTune_BERLIN.hdf5',
            './ExtendedChallenge_Model/FineTune_ISTANBUL.hdf5',
            './ExtendedChallenge_Model/FineTune_MELBOURNE.hdf5',
        ]
    },
}

if __name__ == '__main__':

    for submission_id, data in config.items():
        for competition_id, models in data.items():
            print(f"{submission_id} {competition_id}: {len(models)}")
            for m in models:
                p = f"{BASE_PATH}/{submission_id}/{m}"
                nb_params = 0
                # TODO "a bit" hacky....
                if p.endswith("/"):
                    # https://www.tensorflow.org/guide/checkpoint#manually_inspecting_checkpoints
                    reader = tf.train.load_checkpoint(p)
                    shape_from_key = reader.get_variable_to_shape_map()
                    dtype_from_key = reader.get_variable_to_dtype_map()

                    for k in shape_from_key.keys():
                        if not k.startswith("net/"):
                            continue
                        n_params_ = np.prod(shape_from_key[k]) if len(shape_from_key[k]) > 0 else 0
                        # print(f"{k} {shape_from_key[k]} -> {n_params_}")
                        nb_params += n_params_
                elif p.endswith("hdf5"):
                    from tensorflow_addons.optimizers import LAMB

                    LAMB  # noqa
                    m = tf.keras.models.load_model(p)
                    for w in m.trainable_weights:
                        nb_params += (np.prod(w.shape) if len(w.shape) > 0 else 0)
                else:
                    m = torch.load(p, map_location=torch.device("cpu"))
                    if isinstance(m, dict) and 'model' in m:
                        m = m['model']
                    elif isinstance(m, dict) and 'state_dict' in m:
                        m = m['state_dict']
                    for k, v in m.items():
                        assert isinstance(v, torch.Tensor), p
                        n_params = np.prod(v.size()) if len(v.size()) > 0 else 0
                        nb_params += n_params
                print(f"       {p}: {type(m)}  -> {nb_params}")
