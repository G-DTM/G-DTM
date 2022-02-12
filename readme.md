

Overview of this repository:

    ckpt: This folder contains training models for topic modeling and prediction.

    data: This folder includes the training data set The processed datasets - UK/US/Australia

Installation Instructions:

    Clone Repository: git clone ...

    Create virtual environment (this project runs on Python 3.6): conda create --name gdtm python=3.7

    Activate virtual environment: conda activate gdtm

    Fetch requirements: pip3 install -r requirements.txt

    Run code:
    (1) modeling: python train-model.py --country_name UK --n_topic 60 --num_epoch50 --no_below 1 --no_above 0.8 --param_n 1
    (2) forcasting: python train-forcasting.py --country_name UK --num_epochs 50
Arguments:

        --country_name: Name of country dataset, e.g UK/US/Australia.
        --n_topic: Num of topics.
        --no_below: The lower bound of count for words to keep, e.g 1.
        --no_above: The ratio of upper bound of count for words to keep, e.g 0.3.
