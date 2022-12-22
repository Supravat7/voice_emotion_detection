# Import libraries and modules

from flask import Flask, render_template, request
import numpy as np
from keras.models import load_model
import librosa
from playsound import playsound


app = Flask(__name__)

# Loading crop recommendation model

model = load_model('emotions_classification.h5')



# render home page

@ app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    
    filename = playsound()
    y, sr = librosa.load(filename, duration = 3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

#    eval_features = [eval(mfcc) for x in request.form.values()]
    final_features = [np.array(mfcc)]
    prediction = model.predict(final_features)
    output = prediction[0]
    
    if output == 0:
        return render_template('index.html', prediction='Speech Emotion is fear')
    
    elif output == 1:
        return render_template('index.html', prediction='Speech Emotion is disgust')
    
    elif output == 2:
        return render_template('index.html', prediction='Speech Emotion is happy')
    
    elif output == 3:
        return render_template('index.html', prediction='Speech Emotion is sad')
    
    elif output == 4:
        return render_template('index.html', prediction='Speech Emotion is angry')
    
    elif output == 5:
        return render_template('index.html', prediction='Speech Emotion is pleasant surprise')
    
    elif output == 6:
        return render_template('index.html', prediction='Speech Emotion is neutral')
   
    


if __name__ == '__main__':
    app.run(debug=True)