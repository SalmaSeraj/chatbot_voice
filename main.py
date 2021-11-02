import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

with open("intents.json") as file:
    Corpus = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []

for intent in Corpus["intents"]:
    for pattern in intent["patterns"]:
        words_temp = nltk.word_tokenize(pattern)
        words.extend(words_temp)
        docs_x.append(words_temp)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)
    Train_data = []
    Target_var = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
        words_temp = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in words_temp:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        Train_data.append(bag)
        Target_var.append(output_row)

    Train_data = numpy.array(Train_data)
    Target_var = numpy.array(Target_var)

try:
    model.load("model.tflearn")
except:
    tensorflow.compat.v1.get_default_graph()
    net = tflearn.input_data(shape=[None, len(Train_data[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, len(Target_var[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(Train_data, Target_var, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

#def chat():
 #   print("Start talking with the bot (type quit to stop)!")
  #  while True:
   #     inp = input("You: ")
    #    if inp.lower() == "quit":
     #       break

      #  results = model.predict([bag_of_words(inp, words)])
 #       results_index = numpy.argmax(results)
 #       tag = labels[results_index]

 #       for tg in Corpus["intents"]:
 #           if tg['tag'] == tag:
 #               responses = tg['responses']

#        print(random.choice(responses))

#Voice based assistance
from gtts import gTTS
import os
from pygame import mixer
from tempfile import TemporaryFile

def speak(msg):
    message=gTTS(text=msg, lang='en',slow=False)
    message.save('speak.mp3')
    mixer.init()
    mixer.music.load('speak.mp3')
    mixer.music.play()
    time.sleep(5)
    mixer.quit()
    os.remove('speak.mp3')
    return

import speech_recognition as sr
import time
def listen():
    r=sr.Recognizer()
    with sr.Microphone() as source:
        audio= r.listen(source,phrase_time_limit=10)
        try:
            txt=r.recognize_google(audio)
        except:
            speak("Sorry, Could not hear, can you speak again?")
            time.sleep(2)
            speak("Please speak again")
            audio = r.listen(source, phrase_time_limit=10)
            try:
                txt=r.recognize_google(audio)
            except:
                txt="quit"
    return txt

def chat():
    speak("welcome to travel books,how can I assist you?")
    time.sleep(2)
    while True:
        input=listen()
        if input.lower()=="wait":
            speak("sure")
            time.sleep(5)

        if input.lower()=="quit":
            speak("Thanks for visiting, bye")
            break

        results = model.predict([bag_of_words(input, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in Corpus["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        answer= random.choice(responses)
        speak(answer)

chat()