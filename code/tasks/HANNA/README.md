**Please first [setup the simulator](https://github.com/khanhptnk/hanna-private/tree/master/code)!**

Scripts for reproducing results in [our EMNLP'19 paper](https://arxiv.org/abs/1909.01871) are in the `exp_scripts` directory. You need to run these scripts **inside the `exp_scripts` directory on the Docker image**.

```
~/mount/hanna/code/tasks/HANNA# cd exp_scripts/
```

**IMPORTANT: the evaluation scripts use a batch size of 1 to avoid non-deterministic agent behavior. The scripts will generate results in Tables 9, 10, 11, 12, 13 in Appendix F of the Arxiv paper. To generate results in Tables 3, 4, 5, 6, 8, change the `batch_size` flag in the scripts to 32.**

We provide the pre-trained model of our final agent [here](https://www.dropbox.com/s/6b6yyr6dic6vu2c/hanna_main_pretrained.zip?dl=1).

To learn how to run a script, type:

```
~/mount/hanna/code/tasks/HANNA/exp_scripts# grep '^\# >' ${script_name}.sh
```

For example:

```
~/mount/hanna/code/tasks/HANNA/exp_scripts# grep '^\# >' train_main.sh
# > Train final agent
# > USAGE: bash train_main.sh [device_id]
```

The `device_id` (GPU id) argument is always optional. You can train our final agent as follows
```
~/mount/hanna/code/tasks/HANNA/exp_scripts# bash train_main.sh
```

Training takes about 10 hours on a Titan Xp GPU. After training, you can evaluate the main model as follows
```
~/mount/hanna/code/tasks/HANNA/exp_scripts# bash eval_main.sh
```

To train the LSTM-based agent, you need to switch to the `lstm` branch
```
(on another terminal tab)
$ git checkout lstm
(on the terminal tab where docker is running)
~/mount/hanna/code/tasks/HANNA/exp_scripts# bash train_lstm.sh
~/mount/hanna/code/tasks/HANNA/exp_scripts# bash eval_lstm.sh
```

Results will be saved to a `json` file and evaluated metrics will be printed to the screen. You can also re-compute the metrics using `manual_score.py`
```
$ python manual_score.py ${result_name}.json
```

As the result files are generally very large and hard to view, we also provide you a `debug_tool.py` for better scrutinizing the results
```
$ python debug_tool.py ${result_name}.json
Type 'random' or a task's instr_id to scrutinize result of the task
After a task is loaded, type any information's key (e.g., 'agent_ask') to view it
List of information keys can be found at:
  https://github.com/khanhptnk/hanna/blob/master/code/tasks/HANNA/verbal_ask_agent.py#L36
Steps in GREEN denotes the agent has been to this location before
* denotes help request
+ denotes repeated help request at the same location
>>> random
140786_0 x8F5xyUWy9e find a curtain True
0 ['c9b5c7e60e7c42b48598171d960ec6c4', 0.0, 0.0]
* 1 ['1e40d5ffa75a4f97824d683b755c09ba', 1.5707963267948966, 0.0, 18]
2 (1) ['1e40d5ffa75a4f97824d683b755c09ba', 3.141592653589793, 0.0, 18]
....
>>> agent_ask
0 0
1 1
2 0
3 0
4 0
5 0
....
>>> agent_reason
0 []
1 []
2 ['already_asked']
3 []
....
```
