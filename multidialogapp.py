import os
import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.llms import OpenAI
from multiagent import *
import random
import torch
import config
from googletrans import Translator
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = 
# os.environ['OPENAI_API_KEY'] = 
os.environ["OPENAI_API_KEY"] = config.openaiapihaea



# if st.button('clearcache'):
#     st.cache_resource.clear()


#FUNCTIONS/STATES
@st.cache_resource
def load_model():
    # st.write('loading model- this will happen only once')
    sdmodelpip = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16)
    return sdmodelpip
# sdmodelpip = load_model()


translator = Translator()
def trans(textinput):
    return translator.translate(textinput,src='en',dest='ko').text
    
def transtoeng(textinput):
    return translator.translate(textinput,dest='en').text


if 'chara' not in st.session_state:
    st.session_state.chara = []
def submitchara():
    st.session_state.chara.append(st.session_state.chara_widget)
    st.session_state.chara_widget = ''


if 'talk_history' not in st.session_state:
    st.session_state.talk_history = [[],[]]
simulator = None

if 'name_msg' not in st.session_state:
    st.session_state.name_msg = ['','']

if 'talking_chara' not in st.session_state:
    st.session_state.talking_chara = []
if 'chara_img' not in st.session_state:
    st.session_state.chara_img = ''

def produce_next_conv():
    simulator = st.session_state.get('simulator')
    name, message = simulator.step([username,st.session_state.user_text])
    st.session_state['simulator'] = simulator
    st.session_state.talk_history[0].append(f'\n{name} \:'.upper()+ trans(f' {message}'))
    st.session_state.talk_history[1].append(f'\n{name} \:'.upper()+ f' {message}')

    return name,message

# if 'user_text' not in st.session_state:
st.session_state.user_text = ''
def submit_user_text():
    st.session_state.user_text = transtoeng(st.session_state.user_text_widget)
    produce_next_conv()
    st.session_state.user_text = ''
    st.session_state.user_text_widget = ''


def create_n_get_names():
    names = search_names(st.session_state.getname_widget,num_chara)
    st.session_state.chara = names



######################
##PAGE LAYOUT
st.title("Find the culprit!: Simulation of solving a murder case")
st.title("범인을 찾아라!: 살인사건의 전말...")


if st.button('reload'):
    st.experimental_rerun()

userapi = st.text_input("Your OpenAI api key", type="password")
if userapi == '':
    st.stop()

if userapi:
    os.environ["OPENAI_API_KEY"] = userapi

num_chara = st.slider('Choose Number of characters',0,20,4)


getname = st.text_input('Generate characters with keyword',placeholder='eg,trending, marvel, celebrity show names...',key = 'getname_widget', on_change=create_n_get_names)

chara_input = st.text_input('Add character name', key='chara_widget', on_change=submitchara)

st.session_state.chara

delete_chara = st.selectbox('delete character', ['Choose any character to delete']+st.session_state.chara, label_visibility='collapsed')
if st.button('clear characters'):
    st.session_state.chara = []
    st.experimental_rerun()


select_victim = st.selectbox('Select Victim:',['Select Victim']+st.session_state.chara, key='select_victim_widget')


initialize_but = st.button('Character Initialize')
initializeplace = st.empty()
if initialize_but:
    if select_victim == 'Select Victim':
        initializeplace.warning("Please choose a victim")
    else:
        initializeplace.text('Searching & Creating Character Persona...')
        st.session_state['talking_chara'] = st.session_state.chara.copy()
        st.session_state.talking_chara.remove(st.session_state.select_victim_widget)
        st.session_state.chara_idx = {name:idx for idx,name in enumerate(st.session_state.talking_chara)}
        character_set = set_pipeline(st.session_state.talking_chara,select_victim)
        st.session_state['character_set'] = character_set
        initializeplace.text('Done')
        for i in range(len(st.session_state.character_set['character_descriptions'])):
            st.write(trans('{} ###  Relationship with victim:{}'.format(st.session_state.character_set['character_descriptions'][i],st.session_state.character_set['character_relationships'][i])))



img_qual = st.slider('Image Quality, Diffusion steps',0,50,10)
imginit = st.button('image Initialize')
imginitplace = st.empty()
imgtemp = st.button('image')
imageplace = st.session_state.chara_img
if st.session_state.chara_img != '':
    st.image(st.session_state.chara_img, width=250)  




start_but = st.button('Start/Reset_conv')

transko = st.checkbox('Translate/번역',key='trans_widget')

if transko: 
    st.write('\n'.join(st.session_state.talk_history[0]))    
else:
    st.write('\n'.join(st.session_state.talk_history[1]))


if st.session_state.name_msg[0] in st.session_state.talking_chara:
        idx = st.session_state.chara_idx[st.session_state.name_msg[0]]
        if st.button('move'):
            movement = re.findall(r"\*(.*?)\*",st.session_state.name_msg[1])
            sdmodelpip = load_model()
            st.write(st.session_state.chara_looks[idx] + ':  ' + movement[-1])
            imagenp = action_image_gen(sdmodelpip,img_qual,st.session_state.chara_looks[idx],movement[-1],st.session_state.name_msg[0])
            st.image(imagenp,width=250)
            
        if st.button('Speak'):
            vidresult = get_vid(idx,st.session_state.name_msg[1],st.session_state.chara_sex[idx],st.session_state.chara_img_name[idx])
            if vidresult == 'error':
                st.write(f'{st.session_state.name_msg[0]} refused to speak to you! (error)')
            else:
                # st.write(vidresult)
                st.video(vidresult)

next_but = st.button('Next')

username = st.text_input("Your name")

userprompt = st.text_input("engage in conv",key='user_text_widget',on_change=submit_user_text)


stop_but = st.button('stop')
# st.text('|\n|\n|')

guess_but = st.button('Guess')
if guess_but:
    simulator = st.session_state.get('simulator')
    st.write('Characters have voted the culprit to be...\n',simulator.final_call(st.session_state.talking_chara))

user_guess = st.selectbox('Your Guess:', ['Choose the culprit']+st.session_state.chara)



###########################
#BUTTON/INPUT
if stop_but:
    st.stop()




if start_but:
    st.session_state.talk_history = [[],[]]
    simulator, specified_topic, evidences = run_pipeline(st.session_state.talking_chara,select_victim,st.session_state.character_set)
    # st.write('start:',specified_topic,'evi',evidences)
    st.session_state['simulator'] = simulator
    st.session_state.talk_history[0].append('Detective: '.upper()+trans(specified_topic)+trans(f'  \nEvidences  (only the detective knows this):{evidences}'))
    st.session_state.talk_history[1].append('Detective: '.upper()+specified_topic+f'  \nsEvidences  (only the detective knows this):{evidences}')

    st.experimental_rerun()   


if next_but:
    name, message = produce_next_conv()
    st.session_state.name_msg = [name,message]
    # if name in st.session_state.talking_chara:
    #     idx = st.session_state.chara_idx[name]
    #     if st.button('Speak'):
    #         vidresult = get_vid(idx,message,st.session_state.chara_sex[idx])
    #         if vidresult == 'error':
    #             st.write('error- cannot produce...')
    #         else:
    #             st.video(vidresult)
    st.experimental_rerun()




if userprompt:

    st.experimental_rerun()


if user_guess != 'Choose the culprit':
    culprit = st.session_state.character_set['culprit']
    st.write(f'You chose {user_guess} as the culprit.')
    if culprit == user_guess:
        st.write('Correct! Well done!')
        st.balloons()
    else:
        st.write(f"Wrong! The true culprit is {culprit}")


if delete_chara != 'Choose any character to delete':
    st.session_state.chara.remove(delete_chara)
    st.experimental_rerun() 

if imginit:
    with st.spinner('Getting Character looks'):
        chara_looks, chara_sex = generate_looks_description(st.session_state.talking_chara)
    st.session_state.chara_sex = chara_sex
    st.session_state.chara_looks = chara_looks
    imageinitplace = st.write('Done') 

if imgtemp:
    sdmodelpip = load_model()
    imagepics, chara_images_name= image_gen(sdmodelpip,st.session_state.talking_chara,img_qual,st.session_state.chara_looks)
    st.session_state.chara_img = imagepics
    st.session_state.chara_img_name = chara_images_name
    st.experimental_rerun()
    # st.image('./charaprofileimg')
    # for i in range(2): #len(st.session_state.talking_chara)
    #     st.image(f'image/charaprofileimg{i}')

st.stop()