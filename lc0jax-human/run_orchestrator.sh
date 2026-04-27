export PATH="/root/.cargo/bin:$PATH"
cd /root/lc0jax-human
uv venv .venv --python 3.11
uv pip install 'jax[cpu]'
uv pip install -e .
export PYTHONPATH=$PYTHONPATH:$(pwd)
nohup .venv/bin/python -u scripts/sweep_manager.py --experiments hparam_sweep.jsonl > /root/sweep_hparam.log 2>&1 &
