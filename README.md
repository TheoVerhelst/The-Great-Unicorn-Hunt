# The Great Unicorn Hunt
Advanced Machine Learning coursework, by The Unicorn Hunters

## Setup
Extract `data/train.zip` and `data/test.zip` in `data/`, then create a virtual environment (with virtualenv, or let PyCharm handle it), then
run the following in a terminal with the virtual env activated:

```
pip3 install --upgrade -r requirements.txt
python3 code/dataPrep/prepare.py
```

You can now run the various scripts, for example

```
python3 code/analysis/heatmap.py
```
