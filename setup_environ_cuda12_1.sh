current_dir=$(pwd)
parent_dir=$(dirname "$current_dir")
uv venv --python=3.11
source .venv/bin/activate
uv pip install torch==2.5.0+cu121 -f https://download.pytorch.org/whl/torch
uv pip install k2==1.24.4.dev20250715+cuda12.1.torch2.5.0 -f https://k2-fsa.github.io/k2/cuda.html
uv pip install git+https://github.com/lhotse-speech/lhotse
uv pip install -r requirements.txt
cd $parent_dir
if [ ! -d "icefall" ]; then
  git clone https://github.com/k2-fsa/icefall
fi
cd icefall
uv pip install -r requirements.txt
uv pip install -e .
cd $current_dir
echo "export PYTHONPATH=$parent_dir/icefall:$PYTHONPATH" >> .venv/bin/activate
