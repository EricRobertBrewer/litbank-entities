# litbank-entities

Extract entities from literary texts.

## Install

Download the data set.
```
git clone https://github.com/dbamman/litbank.git ./litbank/
```

Create virtual environment and install dependencies.
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
