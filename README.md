# Machine Learning Prediction of Metal Organic Framework Porosity

A [Flask](https://flask.palletsprojects.com/en/2.2.x/) web app hosting a model produced for the prediction of Metal organic framework (MOF) porosity. This is a web app version of a model which [has been published](https://onlinelibrary.wiley.com/doi/10.1002/anie.202114573).

This can be used as is or as an example of creating a basic flask app for a machine learning model (or any function that takes in two inputs and returns an output)

In order to run this you must first decompress the model files.
We use 7zip as it was the only compression algorithm to compress files to less than github's 25mb limit
You may need to install p7zip first:
```
$ sudo apt install p7zip-full
```
Then extract the files:
```
$ 7z x app/model_M1.7z -oapp
$ 7z x app/model_M2.7z -oapp
$ 7z x app/model_M3.7z -oapp
```
