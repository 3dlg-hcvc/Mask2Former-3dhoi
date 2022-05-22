
module load LIB/CUDNN/8.0.5-CUDA11.1

#  The env command on Shawn's lab machine
conda create -n mask2former python=3.7
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -U opencv-python
pip install git+https://github.com/cocodataset/panopticapi.git
pip install -r requirements.txt

cd mask2former/modeling/pixel_decoder/ops
python setup.py build install