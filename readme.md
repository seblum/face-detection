# Face Detection

## Run Face detection app

```
# run with default settings
poetry run streamlit run app.py
```

Open the IP given in the console in your browser. This will be something like http://localhost:8501.

To start the video, select the use case you want, either mood detection or mask detection, click on the checkbox "Make it run" and start to dance.

## Run Flask Webstreaming

```
poetry run flask --app face-detection/webstreaming --debug run -p 5001

```
