# airbus-challenge

See [Airbus Ship Detection](https://www.kaggle.com/c/airbus-ship-detection) challenge on Kaggle for more details on the given problem

1. Run ```./scripts/download.sh```
2. Make sure ```conda``` is installed

# Comments
- Different models are currently implemented on different branches (i.e. ```dev/george/resnet101_...``` or ```/dev/george/fcn8_...```
- A template file shows how to train a model is ```./training/FCN8s_template.py```
- The main file that does most of the work (i.e. training, checkpoints, printing out live stats) is ```./src/trainer.py```

# To train a model:
1. Duplicate and modify ```training/FCN8s_template.py``` file
2. Start training: ```python -m training.FCN8s_template```

# To execute tests, run:
```python -m unittest discover -v -p "tests_*py"```

# To run in a headless mode:
``` python -u MODULE_OR_CLASS &> out.log &```

This function will run the code and redirect stdout, stderr to file out.log
