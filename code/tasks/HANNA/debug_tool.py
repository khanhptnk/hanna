import os
import sys
import json
from collections import defaultdict
import pprint
import random

pprint._sorted = lambda x:x

from termcolor import colored

filename = sys.argv[1]

with open(filename) as f:
    data_list = json.load(f)

print(colored("Type 'random' or a task's instr_id to scrutinize result of the task", "yellow"))
print(colored("After a task is loaded, type any information's key (e.g., 'agent_ask') to view it", "yellow"))
print(colored('List of information keys can be found at:', 'yellow'))
print('  https://github.com/khanhptnk/hanna/blob/master/code/tasks/HANNA/verbal_ask_agent.py#L36')
print(colored('Steps in ', 'yellow') + colored('GREEN', 'green') + colored(' denotes the agent has been to this location before', 'yellow'))
print(colored("* denotes help request", 'yellow'))
print(colored("+ denotes repeated help request at the same location", 'yellow'))

data_dict = {}
for item in data_list:
    data_dict[item['instr_id']] = item

while True:
    text = input('>>> ')
    try:
        if text == 'random':
            text = random.choice(data_list)['instr_id']
        if '_0' in text:
            example = data_dict[text]
            print(example['instr_id'], example['scan'],
                example['instruction'][0], example['is_success'])
            viewpoints = {}
            ask_points = set()
            for i, item in enumerate(example['agent_pose']):
                if item[0] in viewpoints:
                    message = colored(f'{i} ({viewpoints[item[0]]}) {item}', 'green')
                else:
                    message = f'{i} {item}'
                if i < len(example['agent_ask']) and example['agent_ask'][i] == 1:
                    if item[0] in ask_points:
                        message = '+ '  + message
                    ask_points.add(item[0])
                if i < len(example['agent_ask']) and example['agent_ask'][i] == 1:
                    message = '* ' + message
                print(message)
                viewpoints[item[0]] = i
        else:
            for i, item in enumerate(example[text]):
                print(i, end=' ')
                if isinstance(item, dict):
                    print(json.dumps(item, indent=1))
                else:
                    print(item)
    except KeyError:
        pass
