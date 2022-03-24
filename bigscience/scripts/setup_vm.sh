sudo pip3 uninstall jax jaxlib -y
pip3 install -U pip
pip3 install jax==0.2.25 jaxlib==0.1.74
rm libtpu_tpuv4-0.1.dev*
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*

mkdir -p ~/code
cd ~/code

# Install t5 master version
git clone https://github.com/google-research/text-to-text-transfer-transformer.git
pushd text-to-text-transfer-transformer
pip3 install -e .
popd

git clone https://github.com/bigscience-workshop/t-zero.git
pushd t-zero
# TODO: remove once https://github.com/bigscience-workshop/t-zero/pull/24 is merged
git checkout thomas/update_promptsource_dependency
pip3 install -e ".[seqio_tasks]"
popd

git clone https://github.com/bigscience-workshop/t5x.git
pushd t5x
pip3 install -e ".[bigscience]"
popd

git clone https://github.com/EleutherAI/lm-evaluation-harness.git
pushd lm-evaluation-harness
pip3 install -e .
popd

# TODO: figure if this is actually important
sudo rm /usr/local/lib/python3.8/dist-packages/tensorflow/core/kernels/libtfkernel_sobol_op.so
