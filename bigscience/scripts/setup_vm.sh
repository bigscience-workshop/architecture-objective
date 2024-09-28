sudo pip3 uninstall jax jaxlib -y
pip3 install -U pip
pip3 install jax==0.2.25 jaxlib==0.1.74
rm libtpu_tpuv4-0.1.dev*
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*

mkdir -p ~/code
cd ~/code

# Install t5 first
git clone https://github.com/google-research/text-to-text-transfer-transformer.git
pushd text-to-text-transfer-transformer
pip3 install -e .
popd

git clone https://github.com/bigscience-workshop/promptsource.git
pushd promptsource
git reset e65186c2b8a544de1eb5c283b11b235033b01514 --hard
pip3 install black==21.12b0 # conflicts with streamlit
pip3 install -r requirements.txt
pip3 install --ignore-requires-python -e . #needed because `promptsource` forces the use of python 3.7
popd

git clone https://github.com/bigscience-workshop/architecture-objective.git
pushd t5x
git checkout thomas/test_loss_normalisation
pip3 install -e .
popd

git clone https://github.com/EleutherAI/lm-evaluation-harness.git
pushd lm-evaluation-harness
pip3 install -e .
popd

# TODO: figure if this is actually important
sudo rm /usr/local/lib/python3.8/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so
