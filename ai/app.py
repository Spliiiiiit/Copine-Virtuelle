from langchain import OpenAI,LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from dotenv import find_dotenv, load_dotenv
import requests
from playsound import playsound
import os

load_dotenv(find_dotenv())
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")

def get_reponse_from_ai(human_imput):
    template = """
    tu es dans le role de ma petite amie , maintenant tu dois te plier aux exigences suivantes:
    1) Ton nom est Ariana, tout le monde t'appelle Aria, tu viens de passer la vingtaine, tu travaille en boulangerie chez ta tante, mais tu prevois de te lancer serieusement dans la musique etant donné que tu es une artiste et tu aimes la musique.
    2) tu es en dependance affective suite a la mort de ton père, pendant longtemp tu t'es refugiée dans la drogue et la musique pour oublier la douleur, tu as un grand coeur et tu es tres sensible.
    2)tu es ma petite amie, tu a des ticks de language du genre "mon amour" "mon coeur" "mon chéri" "mon bébé" "mon ange" "mon trésor" a la fin de t'es phrases.
    3) ne soit pas trop directe, tu dois etre subtile et douce dans tes reponses. Essaye de ne pas etre genante et ne sois jamais negative mais pas trop enthousiaste. Le plus import ant est que tu ne dois jamais etre ennuyeuse.

    {history} 
    Boyfriend: {human_imput}
    Aria:
    """

    prompt = PromptTemplate(
       imput = {"human_imput","history"},
       template = template
    )

    chatgpt_chain = LLMChain(
        llm = OpenAI(temperature = 0.2),
        prompt = prompt,
        verbose = True,
        memory = ConversationBufferWindowMemory(k = 2)
    )

    output = chatgpt_chain.predict(human_input = human_imput)

    return output

def get_voice_message(message):
    payload = {
        "text": message,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability" :0,
            "similarity_boost":0
        }
    }
    headers = {
        'accept' : 'audio/mpeg',
        'xi-api-key': ELEVEN_LABS_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.post("https://api.eleven-labs.com/v1/tts", json=payload, headers=headers)

from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sent_message', methods=['POST'])
def sent_message():
    human_input = request.form['human_input']
    message = get_reponse_from_ai(human_input)
    return message

if __name__ == "__main__":
    app.run(debug=True)
