from langchain import PromptTemplate
import re
import random
import numpy as np
import tenacity
from typing import List, Dict, Callable
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
import os


word_limit = 50


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.message_history = ["Here is the conversation so far."]
        self.prefix = f"{self.name}:"

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self,user=[0,0]) -> tuple[str, str]:
        if user[1]:
            name = user[0]
            message = user[1]

            for receiver in self.agents:
                receiver.receive(name, message)
            self._step += 1
            print('user talks')
            print(self.agents[0].message_history)

            return name, message
        
        else:
            # 1. choose the next speaker
            speaker_idx = self.select_next_speaker(self._step, self.agents)
            speaker = self.agents[speaker_idx]
            print('agent talks')
            # 2. next speaker sends message
            message = speaker.send()

        # 3. everyone receives message
            for receiver in self.agents:
                receiver.receive(speaker.name, message)

        # 4. increment time
            self._step += 1
            print(self.agents[0].message_history)   
            return speaker.name, message
            
            
    # def return_firstagenthist(self):
    #     return str(self.agents[0].message_history)
            

class BiddingDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        bidding_template: PromptTemplate,
        model: ChatOpenAI,
    ) -> None:
        super().__init__(name, system_message, model)
        self.bidding_template = bidding_template
        
    def bid(self) -> str:
        """
        Asks the chat model to output a bid to speak
        """
        prompt = PromptTemplate(
            input_variables=['message_history', 'recent_message'],
            template = self.bidding_template
        ).format(
            message_history='\n'.join(self.message_history),
            recent_message=self.message_history[-1])
        bid_string = self.model([SystemMessage(content=prompt)]).content
        return bid_string
        

def generate_a_character_description(character_name):
    character_specifier_prompt = [SystemMessage(
    content="You can add detail to the description."), HumanMessage(content=
            f"""Here is the topic for the argument:
            Please reply with a creative description of {character_name}, with desciption lifestyle before the death, in 50 words or less.
            Do not add anything else.""")]
    character_description = ChatOpenAI(temperature=1.0)(character_specifier_prompt).content
    return character_description


def generate_character_description(character_name,game_description):
    player_descriptor_system_message = SystemMessage(
        content="You can add detail to the description of each suspects.")
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(content=
            f"""{game_description}
            Please reply with a creative description of the suspect, {character_name}, in {word_limit} words or less, that emphasizes their personalities. 
            Speak directly to {character_name}.
            Do not add anything else."""
            )
    ]
    character_description = ChatOpenAI(temperature=1.0)(character_specifier_prompt).content
    return character_description

def generate_character_relationship(character_name,dead,game_description):
    character_rela_prompt = [SystemMessage(content="You can add detail to the description"),
                              HumanMessage(content=f'''{game_description}
                                           Please reply with creative description of relationship between the suspect,{character_name} and {dead}, in 20 words or less.
                                           If there is no discernible relationship, come up with a fictional relationship.
                                           Do not add anything else.''')] #Here is the topic for the fictional argument: {topic} #Speak directly to {character_name}. #Start sentence with "Your relationship with {dead} is " 
    character_relationship = ChatOpenAI(temperature=1.0)(character_rela_prompt).content.replace('Fictional relationship','').replace(':','').replace('-','').replace('Fictional Relationship','')
    if 'Sorry' in character_relationship:
        return f"There is no publicly known discernible relationship between {character_name} and {dead}"
    return character_relationship



def generate_character_header(character_name, character_description,character_relationship,dead,game_description,topic,culprit=''):
    return f"""{game_description}
Your name is {character_name}.
{culprit}You are a murder suspect.
Your description is as follows: {character_description}
You are arguing on the topic: {topic}.
Here is description of your relationship with {dead}: {character_relationship}
Your goal is to be as creative as possible and make the everyone think you are innocent and name a culprit at the end of the argument.
"""

def generate_character_system_message(character_name, character_header,topic):
    return SystemMessage(content=(
    f"""{character_header}
You will speak in the style of {character_name}, and exaggerate their personality.
You will come up with creative ideas related to {topic}.
Try your best to show you are innocent while raising suspicion of other suspects.
Do not say the same things over and over again.
Speak in the first person from the perspective of {character_name}
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {character_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
    """
    ))


class BidOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return 'Your response should be an integer delimited by angled brackets, like this: <int>.'     
    
bid_parser = BidOutputParser(
    regex=r'<(\d+)>', 
    output_keys=['bid'],
    default_output_key='bid')

def generate_character_bidding_template(character_header):
    bidding_template = (
    f"""{character_header}

```
{{message_history}}
```

On the scale of 1 to 10, where 1 is not necessary and 10 is extremely necessary, from the following message rate how necessary for you to speak up to is to clear your suspicion of murder.

```
{{recent_message}}
```

{bid_parser.get_format_instructions()}
Do nothing else.
    """)
    return bidding_template


@tenacity.retry(stop=tenacity.stop_after_attempt(2),
                    wait=tenacity.wait_none(),  # No waiting time between retries
                    retry=tenacity.retry_if_exception_type(ValueError),
                    before_sleep=lambda retry_state: print(f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
                    retry_error_callback=lambda retry_state: 0) # Default value when all retries are exhausted
def ask_for_bid(agent) -> str:
    """
    Ask for agent bid and parses the bid into the correct format.
    """
    bid_string = agent.bid()
    bid = int(bid_parser.parse(bid_string)['bid'])
    return bid


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)
        
    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)
    
    print('Bids:')
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f'\t{agent.name} bid: {bid}')
        if i == idx:
            selected_name = agent.name
    print(f'Selected: {selected_name}')
    print('\n')
    return idx


def set_pipeline(character_names,dead):

    culprit = character_names[random.randint(0, len(character_names)-1)]
    topic = f"Who murdered {dead}?" 
    
    dead_description = generate_a_character_description(dead)
    game_description = f"""Here is the topic for the argument: {topic}.
Description of {dead}:{dead_description} 
The suspects are: {', '.join(character_names)}. One of them is a culprit"""

    character_descriptions = [generate_character_description(character_name,game_description) for character_name in character_names]
    character_relationships = [generate_character_relationship(character_name,dead,game_description) for character_name in character_names]
    character_headers = []
    for character_name, character_description, character_relationship in zip(character_names, character_descriptions,character_relationships):
        if culprit == character_name:
            character_headers.append(generate_character_header(character_name, character_description,character_relationship,dead,game_description,topic,culprit='You are the culprit. '))
        else:
            character_headers.append(generate_character_header(character_name, character_description,character_relationship,dead,game_description,topic))

    character_system_messages = [generate_character_system_message(character_name, character_headers,topic) for character_name, character_headers in zip(character_names, character_headers)]

    character_bidding_templates = [generate_character_bidding_template(character_header) for character_header in character_headers]
    
    character_set = {
        'topic': topic,
        'game_description':game_description,
        'character_bidding_templates': character_bidding_templates,
        'character_system_messages': character_system_messages,
        'character_relationships': character_relationships
    }
    
    return culprit,character_set


def run_pipeline(character_names,dead,character_set):
    game_description = character_set['game_description']
    character_bidding_templates = character_set['character_bidding_templates']
    character_system_messages = character_set['character_system_messages']
    character_relationships = character_set['character_relationships']

    detective_header = f"""{game_description}.
    You are Detective trying to investigate on the case.
    You have the knowledge of the evidences of the case.
    There is no more evidence to be collected beyond this argument.
    Interrogate the suspects with the knowledge.
    You may look out for contradictions in suspects statements.
    You have to decide on a culprit before 30 turns of conversations.
    Your goal is to be as creative as possible to gather as much information as possible to determine the culprit who is amongst the suspects:{', '.join(character_names)}"""

    detective_system_message = SystemMessage(content=(f"""{detective_header}
    Try to make suspects argue against one another.
    Do not say the same things over and over again.
    Speak in the first person from the perspective of a detective.
    For describing your own body movements, wrap your description in '*'.
    Do not change roles!
    Do not speak from the perspective of anyone else!
    Speak only from the perspective of detective.
    Stop speaking the moment you finish speaking from your perspective.
    Never forget to keep your response to {word_limit} words!
    Do not add anything else.""" ))

    detective_bidding_template = (
        f"""{detective_header}

    ```
    {{message_history}}
    ```

    On the scale of 1 to 8, where 1 is not necessary and 10 is extremely necessary,from the following message rate how much is your envolvement necessary to interrogate and lead the argument to determine the culprit.

    ```
    {{recent_message}}
    ```

    {bid_parser.get_format_instructions()}
    Do nothing else.
        """)


    characters = []
    for character_name, character_system_message, bidding_template in zip(character_names, character_system_messages, character_bidding_templates):
        characters.append(BiddingDialogueAgent(
            name=character_name,
            system_message=character_system_message,
            model=ChatOpenAI(temperature=1),
            bidding_template=bidding_template,
        ))

    characters.append(BiddingDialogueAgent(
            name='Detective',
            system_message=detective_system_message,
            model=ChatOpenAI(temperature=0.9),
            bidding_template=detective_bidding_template,
        ))
    simulator = DialogueSimulator(
    agents=characters,
    selection_function=select_next_speaker
)
    topic_specifier_prompt = [
        SystemMessage(content="You can make a task more specific."),
        HumanMessage(content=
            f"""{game_description}
            
            You are the detective acting as moderator.
            Please make the argument topic more specific. 
            Frame the argumet topic as a problem to be solved.
            Be creative and imaginative.
            Please reply with the specified topic in {word_limit} words or less. 
            Speak directly to the suspects: {character_names}.
            Do not add anything else."""
            )
    ]
    specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content + '''\nSuspects relationships with {}: {}'''.format(dead,'\n'.join(character_relationships))
    simulator.reset('Detective', specified_topic)
    return simulator, specified_topic

# def generate_topic_specifier(character_names,dead,character_relationships):
#     topic_specifier_prompt = [
#         SystemMessage(content="You can make a task more specific."),
#         HumanMessage(content=
#             f"""{game_description}
            
#             You are the detective acting as moderator.
#             Please make the argument topic more specific. 
#             Frame the argumet topic as a problem to be solved.
#             Be creative and imaginative.
#             Please reply with the specified topic in {word_limit} words or less. 
#             Speak directly to the suspects: {character_names}.
#             Do not add anything else."""
#             )
#     ]
#     specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content + f'''\nSuspects relationships with {dead}: {' '.join(character_relationships)}'''
#     return specified_topic


# print(f"(Detective Moderator): {specified_topic}")
# print('\n')


# name, message = simulator.step()
# print(f"({name}): {message}")

