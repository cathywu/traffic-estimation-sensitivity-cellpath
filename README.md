# traffic-estimation-sensitivity-cellpath

## Setup

1. Pull [traffic-estimation-wardrop](https://github.com/jeromethai/traffic-estimation-wardrop) and follow the setup instructions.
1. Pull [traffic-estimation](https://github.com/cathywu/traffic-estimation) and follow the setup instructions.
1. Copy `config.py.template` to `config.py` and add the respective paths.

## Running

Sample command

    python main.py main.py --log INFO --NLP 10 --NL 0 --NS 0 --NB 50 --solver LS --model UE --method BB --use_L
