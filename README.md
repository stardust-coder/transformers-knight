# Transformer-Knight🐎

Towards fast inference of softmax attention Transformer.

## Setup
1. Prepare singularity environment.
2. Build sif.
```
singularity build --bind $PWD:/mnt --fakeroot knight.sif knight.def
```
3. run
```
singularity shell --nv knight.sif
python main.py
```

## Temporal solution
```
#Clone original repo.
git clone https://github.com/huggingface/transformers.git

# refer to two files in /tmp/

# Run main.py.
bash run.sh
```

## Files modified from official repo of transformers
1. transformer-knight/transformers/src/transformers/integrations/sdpa_attention.py
    - [original implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py)
2. transformer-knight/transformers/src/transformers/models/llama/modeling_llama.py
    - [original implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py)

## TODO
- Gaussian random features include too big values and too small values, resulting in many zero division errors.
- Gaussian random features are less efficient, see DiJiang for example.