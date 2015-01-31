# traffic-estimation-sensitivity-cellpath

## Setup

1. Pull [traffic-estimation-wardrop](https://github.com/jeromethai/traffic-estimation-wardrop) and follow the setup instructions.
1. Pull [traffic-estimation](https://github.com/cathywu/traffic-estimation) and follow the setup instructions.
1. Copy `config.py.template` to `config.py` and add the respective paths.

## Running

Sample command

    python main.py --log INFO --output --NLP 10 --NL 0 --nrow 4 --solver CS --NB 50 --ncol 8 --nodroutes 15 --model UE --NS 0 --method BB --sparse --use_L
