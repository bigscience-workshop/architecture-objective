sudo pip3 uninstall jax jaxlib -y
pip3 install -U pip
pip3 install jax==0.2.25 jaxlib==0.1.74
rm libtpu_tpuv4-0.1.dev*
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*

### ...
##pip3 install tensorflow==2.7.0
#rm tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl
#gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/tensorflow/tf-2-7-0/tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl .
#pip3 install tensorflow-2.7.0-cp38-cp38-linux_x86_64.whl tensorflow-text==2.7.0

mkdir -p ~/code
cd ~/code

# Install t5 first
git clone https://github.com/google-research/text-to-text-transfer-transformer.git
pushd text-to-text-transfer-transformer
pip3 install -e .
popd
#rm -rf text-to-text-transfer-transformer
#git clone https://github.com/thomasw21/text-to-text-transfer-transformer.git
#pushd text-to-text-transfer-transformer
#git checkout fix_prefix_lm_obj
#pip3 install -e .
#popd

git clone https://github.com/bigscience-workshop/promptsource.git
pushd promptsource
git checkout thomas/t5x
pip3 install -r requirements.txt
pip3 install --ignore-requires-python -e . #needed because `promptsource` forces the use of python 3.7
popd

#rm -rf t5x
git clone https://github.com/bigscience-workshop/t5x.git
pushd t5x
git checkout thomas/prefix_lm_add_token
pip3 install -e .
popd


# TODO: figure if this is actually important
sudo rm /usr/local/lib/python3.8/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so

## TODO: figure why I need this
##   This is probably linked to `use_custom_packing_ops=True`. Just set it to False and we're good to go
#pip3 install tensor2tensor

## Needed for profiling to work apparently
#pip3 install tbp-nightly
