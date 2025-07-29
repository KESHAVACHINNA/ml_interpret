setup:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	pylint --disable=R,C,W1203 app.py

run:
	. .venv/bin/activate && streamlit run app.py

all: install lint run
