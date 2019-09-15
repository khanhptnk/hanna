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

total_ask = 0
cnt_ask = 0

total_nav = 0
cnt_nav = 0
num_actions_per_loc = []

data_dict = {}
for item in data_list:
    data_dict[item['instr_id']] = item

    nav_action_dict = defaultdict(set)
    ask_action_dict = defaultdict(set)
    ask_points = set()
    for a, b, c, d, e in zip(item['agent_pose'], item['agent_ask'],
            item['agent_nav'], item['instruction'], item['teacher_nav']):
        total_nav += 1
        viewpoint = a[0]
        key = viewpoint + ' ' + d
        if key in nav_action_dict:
            cnt_nav += c in nav_action_dict[key] and c != e
        nav_action_dict[key].add(c)
        if b == 1:
            total_ask += 1
            if key in ask_action_dict:
                cnt_ask += b in ask_action_dict[key]
            ask_action_dict[key].add(b)
    for v in nav_action_dict.values():
        num_actions_per_loc.append(len(v))

print('%.2f' % (cnt_ask / total_ask * 100))
print('%.2f' % (cnt_nav / total_nav * 100))
print('%.2f' % (sum(num_actions_per_loc) / len(num_actions_per_loc)))


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
