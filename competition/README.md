## The competition.

### You are here
```
├── competition
│   ├── prepare_test_data       <-- create test manifests and package test data
│   ├── scorecomp               <-- competition scoring
│   ├── static_data_preparation <-- static data pipeline (relevant for participants only if they want to create their own static data)
│   └── submission              <-- create submissions

```

### Get your hands dirty!

```
conda activate t4c
export PYTHONPATH=$PWD
python competition/prepare_test_data/prepare_test_data.py --help
python competition/submission/submission.py --help
python competition/scorecomp/scorecomp.py --help
```
