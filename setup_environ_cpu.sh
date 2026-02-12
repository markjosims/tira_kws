current_dir=$(pwd)
parent_dir=$(dirname "$current_dir")
uv venv --python=3.10
source .venv/bin/activate
uv pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install k2==1.24.4.dev20251118+cpu.torch2.9.1 -f https://k2-fsa.github.io/k2/cpu.html
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