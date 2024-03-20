conda_path=$(echo $(conda info | grep 'envs dir') | cut -d ':' -f 2 | xargs)
PYTHONPATH=$PYTHONPATH:$conda_path/mgltools/MGLToolsPckgs
