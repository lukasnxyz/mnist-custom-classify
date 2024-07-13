### MNIST Inference 
Built this project after self studying the fundamentals of machine learning and deep learning, bottom-up. 
I then started learning pytorch, followed karpathy's tutorial's and learnpytorch.io and wanted to built a
my own "production" project applying machine learning. This is the best/first idea I could come up with.

![example run gif](data/example_run.gif)

#### Quick start
You need torch, numpy, and tkinter or flask depending if you want to run the app or webapp version. Also 
make sure you unzip the mnist data and then add the test into the train file because the splitting happens 
in the notebook.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

```bash
python3 run.py
```
or
```bash
cd app/
python3 backend.py
```
