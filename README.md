## Project Structure
```
│   .gitignore 
│   README.md
│   requirements.txt
│   test.py
│   utils.py
│   __init__.py
│       
│
├───data_loading
│       data_split.py
│       __init__.py
│
├───inference
│       Dockerfile
│       prediction.py
│       __init__.py
│
│
└───training
        Dockerfile
        train.py
        __init__.py
```

## Model

The model is just a logistic regression but written in torch. There is no real need for more complex model, since the dataset is almost linearly separable and it already has close to perfect accuracy.

## Running the Project

#### Data Generation

Run `python data_loading/data_split.py` to load the Iris dataset. This will create data folder in the working directory with training and inference sets.

#### Training

Run training either by `python training/train.py` or in a Docker container. For the second option:
- first create a Docker image by running
```
docker build -f ./training/Dockerfile -t training_image .
```
- then create a container by
```
docker run -dit training_image
```
- to get the model from the container run
```
docker cp <container_id>:/app/model/logistic.pth ./model
```

#### Unit Tests
To run unit test script just execute `python -m unittest test.py`. Alternatively, these unit tests are automatically run in the inference container and their results can be seen in the log file from that container.

#### Inference

For inference either run `python inference/prediction.py` or
- create a docker image
```
docker build -f ./inference/Dockerfile -t inference_image .
```
- then get a running container
```
docker run -dit inference_image
```
- to get the results from the container
```
docker cp <container_id>:/app/results/res.csv ./results
```
Model accuracy is saved in the log file.

#### Logging

All the logs are saved in `history.log` file. If the model is run in a container, the file can be saved locally by running
```
docker cp <container_id>:/app/history.log ./<log_name>.log
```

