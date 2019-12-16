# airbus-challenge

1. Run ```./scripts/download.sh```
2. Make sure ```conda``` is installed

# To execute tests, run:
```python -m unittest discover -v -p "tests_*py"```

# To run in a headless mode:
``` python -u MODULE_OR_CLASS &> out.log &```

This function will run the code and redirect stdout, stderr to file out.log
