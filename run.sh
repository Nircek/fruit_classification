#!/bin/bash

[ ! -f ok ] && {
    [ ! -f virtualenv.pyz ] && wget https://bootstrap.pypa.io/virtualenv.pyz;
    [ ! -f .venv/bin/activate ] && python3 virtualenv.pyz .venv;
    . .venv/bin/activate;
    pip install -r requirements.txt || {
         echo -e "\n\n\nThere was an error in installation...\n\n\n";
         exit;
    }
    deactivate;
    echo -e "\n\n\nVirtualenv created and dependencies installed.\n\n\n";
    touch ok;
}

. .venv/bin/activate
python3 app.py
