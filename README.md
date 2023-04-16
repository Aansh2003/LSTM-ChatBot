# LSTM Based NLU chatbot with randomized responses

This is a chatbot designed to classify inputs and generate apt outputs using Natural Language Processing techniques. This project is currently active under development and will be updated from time to time.

## Installing dependencies

Install pandas and numpy for dataframe and mathematical features
```
pip3 install pandas
pip3 install numpy
```

Install PyTorch for model support
```
pip3 install torch torchvision torchaudio
```

Install keras for preprocessing
```
pip3 install keras
pip3 install Keras-Preprocessing
```

Installing BeautifulSoup for HTML parsing and JamSpell for spell-check
```
pip3 install beautifulsoup4
pip3 install jamspell
```
All required dependencies have now been installed.

### Additional dependencies for flask server deployment
```
pip3 install Flask
```

## Running the model from command line
Type `python3 out.py` and wait a couple of seconds till it shows
>Started...  
>user:

Here is an example output:
![Screenshot from 2023-04-12 20-18-48](https://user-images.githubusercontent.com/96300383/231521115-61341907-e2fd-4901-8d8f-1c510d9d9009.png)

## Running the model using a flask server
Type `python3 flask_test.py` and wait till the server has been deployed. A link will be displayed like the following
> Running on http://127.0.0.1:5000   

Copy the link at paste it on your browser to deploy the website.
Here is an example output:
![image](https://user-images.githubusercontent.com/96300383/232273757-b65c5486-49a1-4aa4-a2f6-50efea661fc6.png)

Currently working on a better deployment method for the model with a proper front-end and also further increasing the number of classes and accuracy of the model and also adding numerous other responses.
