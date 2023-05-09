import os
import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
from langchain.llms import OpenAI
from multiagent import *
import random
import config


# os.environ["HUGGINGFACEHUB_API_TOKEN"] = 
# os.environ['OPENAI_API_KEY'] = 
os.environ["OPENAI_API_KEY"] = config.openaiapihaea

# llm = OpenAI(model_name='gpt-3.5-turbo',temperature=0) 





#FUNCTIONS/STATES

if 'chara' not in st.session_state:
    st.session_state.chara = []
def submitchara():
    st.session_state.chara.append(st.session_state.chara_widget)
    st.session_state.chara_widget = ''

if 'talk_history' not in st.session_state:
    st.session_state.talk_history = []
simulator = None
culprit = None

# def next_but_func():
#     simulator = st.session_state.get('simulator')
#     name, message = simulator.step([username,st.session_state.user_text])
#     st.session_state.talk_history.append(f'\n{name} \:'.upper()+ f' {message}')
    

# if 'user_text' not in st.session_state:
st.session_state.user_text = ''
def submit_user_text():
    st.session_state.user_text = st.session_state.user_text_widget
    simulator = st.session_state.get('simulator')
    name, message = simulator.step([username,st.session_state.user_text])
    st.session_state['simulator'] = simulator
    st.session_state.talk_history.append(f'\n{name} \:'.upper()+ f' {message}')
    st.session_state.user_text = ''
    st.session_state.user_text_widget = ''



# name = st.text_input('Name')
# if not name:
#   st.warning('Please input a name.')
#   st.stop()
# st.success('Thank you for inputting a name.')
    




##PAGE LAYOUT
st.title("It's open")
dead = st.text_input('Dead chara name')
if dead:
    st.write(f'You have chosen {dead} to be dead.')

chara_input = st.text_input('character name', key='chara_widget', on_change=submitchara)

st.session_state.chara

initialize_but = st.button('Initialize')

start_but = st.button('Start/Reset_conv')

st.write('\n'.join(st.session_state.talk_history))

username = st.text_input("Your name")

userprompt = st.text_input("engage in conv",key='user_text_widget',on_change=submit_user_text)

next_but = st.button('Next')

stop_but = st.button('stop')
#BUTTON/INPUT

if stop_but:
    st.stop()


if initialize_but:

    culprit, character_set = set_pipeline(st.session_state.chara,dead)
    st.session_state['simulators'] = []
    st.session_state['cur_id'] = 0
    character_set['id'] = st.session_state['cur_id']
    st.session_state['simulators'].append(character_set)
    st.session_state['cur_id'] += 1

if start_but:
    st.session_state.talk_history = []
    simulator, specified_topic = run_pipeline(st.session_state.chara,dead,st.session_state.simulators[0])
    st.session_state['simulator'] = simulator
    st.session_state.talk_history.append('Detective: '.upper()+specified_topic)
    st.experimental_rerun()   
    # st.write('Detective: '+specified_topic)

if next_but:
    simulator = st.session_state.get('simulator')
    name, message = simulator.step([username,st.session_state.user_text])
    st.session_state['simulator'] = simulator
    st.session_state.talk_history.append(f'\n{name} \:'.upper()+ f' {message}')
    # st.write(simulator.return_firstagenthist())
    st.experimental_rerun()
    
    
    # st.session_state.user_text = ''
    # with placeholder:
    #     st.write('\n'.join(st.session_state.talk_history))
if userprompt:

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


