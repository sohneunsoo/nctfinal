import os
import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.llms import OpenAI
from multiagent import *
import random
import config
from googletrans import Translator

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = 
# os.environ['OPENAI_API_KEY'] = 
os.environ["OPENAI_API_KEY"] = config.openaiapihaea

# llm = OpenAI(model_name='gpt-3.5-turbo',temperature=0) 





#FUNCTIONS/STATES
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


# def next_but_func():
#     simulator = st.session_state.get('simulator')
#     name, message = simulator.step([username,st.session_state.user_text])
#     st.session_state.talk_history.append(f'\n{name} \:'.upper()+ f' {message}')
    

def produce_next_conv():
    simulator = st.session_state.get('simulator')
    name, message = simulator.step([username,st.session_state.user_text])
    st.session_state['simulator'] = simulator
    st.session_state.talk_history[0].append(f'\n{name} \:'.upper()+ trans(f' {message}'))
    st.session_state.talk_history[1].append(f'\n{name} \:'.upper()+ f' {message}')

# if 'user_text' not in st.session_state:
st.session_state.user_text = ''
def submit_user_text():
    st.session_state.user_text = transtoeng(st.session_state.user_text_widget)
    produce_next_conv()
    st.session_state.user_text = ''
    st.session_state.user_text_widget = ''




# name = st.text_input('Name')
# if not name:
#   st.warning('Please input a name.')
#   st.stop()
# st.success('Thank you for inputting a name.')
    




##PAGE LAYOUT
st.title("It's open")

userapi = st.text_input("Your api key", type="password")
if userapi:
    os.environ["OPENAI_API_KEY"] = userapi


transko = st.checkbox('Translate/번역',key='trans_widget')

# dead = st.text_input('Victim character name')
# if dead:
#     st.write(f'You have chosen {dead} to be dead.')

chara_input = st.text_input('character name', key='chara_widget', on_change=submitchara)



st.session_state.chara

if st.session_state.chara == []: 
    label_visibility = 'hidden'
else:
    label_visibility = 'visible'

delete_chara = st.selectbox('', ['Choose a character to delete']+st.session_state.chara, label_visibility=label_visibility)
if st.button('clear characters'):
    st.session_state.chara = []
    st.experimental_rerun()


select_victim = st.selectbox('Select Victim:',st.session_state.chara)


initialize_but = st.button('Initialize')

start_but = st.button('Start/Reset_conv')

if transko:
    st.write('\n'.join(st.session_state.talk_history[0]))
else:
    st.write('\n'.join(st.session_state.talk_history[1]))

next_but = st.button('Next')

username = st.text_input("Your name")

userprompt = st.text_input("engage in conv",key='user_text_widget',on_change=submit_user_text)


stop_but = st.button('stop')

guess_but = st.button('Guess')
if guess_but:
    simulator = st.session_state.get('simulator')
    st.write('Characters have voted the culprit to be...\n',simulator.final_call(st.session_state.talking_chara))

user_guess = st.selectbox('Your Guess:', ['Choose the culprit']+st.session_state.chara)


#BUTTON/INPUT
if stop_but:
    st.stop()


if initialize_but:
    st.session_state['talking_chara'] = st.session_state.chara.copy()
    st.session_state.talking_chara.remove(select_victim)
    character_set = set_pipeline(st.session_state.talking_chara,select_victim)
    # st.session_state['simulators'] = []
    # st.session_state['cur_id'] = 0
    # character_set['id'] = st.session_state['cur_id']
    st.session_state['character_set'] = character_set
    # st.session_state['cur_id'] += 1
    st.write('Done')

if start_but:
    st.session_state.talk_history = [[],[]]
    simulator, specified_topic, evidences = run_pipeline(st.session_state.talking_chara,select_victim,st.session_state.character_set)
    st.session_state['simulator'] = simulator
    st.session_state.talk_history[0].append('Detective: '.upper()+trans(specified_topic)+trans(f'\nEvidences (only the detective knows this):{evidences}'))
    st.session_state.talk_history[1].append('Detective: '.upper()+specified_topic+f'\nEvidences (only the detective knows this):{evidences}')

    st.experimental_rerun()   
    # st.write('Detective: '+specified_topic)

if next_but:
    produce_next_conv()
    # st.write(simulator.return_firstagenthist())
    st.experimental_rerun()
    
    
    # st.session_state.user_text = ''
    # with placeholder:
    #     st.write('\n'.join(st.session_state.talk_history))
if userprompt:

    st.experimental_rerun()



if user_guess != 'Choose the culprit':
    culprit = st.session_state.character_set['culprit']
    st.write(f'You chose {user_guess} as the culprit.')
    if culprit == user_guess:
        st.write('Correct! Well done!')
    else:
        st.write(f"Wrong! The true culprit is {culprit}")


if delete_chara != 'Choose a character to delete':
    st.session_state.chara.remove(delete_chara)
    st.experimental_rerun() 


# def submit_user_text():
#     st.write(st.session_state.user_text)
#     st.write(st.session_state.user_name)
#     # name, message = simulator.step([st.session_state.user_name,st.session_state.user_text])
#     # st.session_state.talk_history.append(f'\n{name} \: {message}')
#     # st.write('\n'.join(st.session_state.talk_history))

# with st.form(key='user_text'):
#     user_name_input = st.text_input("Your name", key='user_name')
#     user_text_input = st.text_input('Join the conversation',key='user_text')
#     submit_button = st.form_submit_button(label='engage',on_click=submit_user_text)


