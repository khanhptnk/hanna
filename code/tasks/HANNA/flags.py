import argparse

def make_parser():
   parser = argparse.ArgumentParser()

   parser.add_argument('-config_file', type=str,
        help='configuration file')
   parser.add_argument('-load_path', type=str,
        help='path to load a pretrained model')
   parser.add_argument('-data_prefix', type=str,
        help='data file prefix')
   parser.add_argument('-exp_name', type=str,
        help='name of the experiment')
   parser.add_argument('-seed', type=int,
        help='random seed')
   parser.add_argument('-data_dir', type=str,
        help='data directory')
   parser.add_argument('-img_features', type=str,
        help='path to pretrained image embeddings')
   parser.add_argument('-img_feature_size', type=int, default=2048,
        help='image embedding size')
   parser.add_argument('-max_instr_len', type=int,
        help='maximum input instruction length')
   parser.add_argument('-batch_size', type=int,
        help='batch size (both training and evaluation)')
   parser.add_argument('-train_episode_len', type=int,
        help='maximum number of actions per epsiode')
   parser.add_argument('-eval_episode_len', type=int,
        help='maximum number of actions per epsiode')
   parser.add_argument('-word_embed_size', type=int,
        help='word embedding size')
   parser.add_argument('-action_embed_size', type=int,
        help='navigation action embedding size')
   parser.add_argument('-ask_embed_size', type=int,
        help='ask action embedding size')
   parser.add_argument('-hidden_size', type=int,
        help='number of LSTM hidden units')
   parser.add_argument('-attention_heads', type=int,
        help='Number of attention heads')
   parser.add_argument('-num_layers', type=int,
        help='Number of transformer layers')
   parser.add_argument('-dropout_ratio', type=float,
        help='dropout probability')
   parser.add_argument('-nav_feedback', type=str,
        help='navigation training method (deprecated)')
   parser.add_argument('-ask_feedback', type=str,
        help='navigation training method (deprecated)')
   parser.add_argument('-lr', type=float,
        help='learning rate')
   parser.add_argument('-weight_decay', type=float,
        help='L2-regularization weight')
   parser.add_argument('-n_iters', type=int,
        help='number of training iterations (batches)')
   parser.add_argument('-min_word_count', type=int,
        help='minimum word count cutoff when building vocabulary')
   parser.add_argument('-split_by_spaces', type=int,
        help='split word by spaces (always set true)')
   parser.add_argument('-start_lr_decay', type=int,
        help='iteration to start decaying learning rate')
   parser.add_argument('-lr_decay_rate', type=float,
        help='learning rate decay rate')
   parser.add_argument('-decay_lr_every', type=int,
        help='number of iterations between learning rate decays')
   parser.add_argument('-log_every', type=int,
        help='number of iterations between information loggings')
   parser.add_argument('-loc_embed_size', type=int,
        help='viewpoint location embedding')

   parser.add_argument('-external_main_vocab', type=str,
        help='provide a different vocab file')

   # Advisor
   parser.add_argument('-advisor', type=str,
        help="type of advisor ('direct' or 'verbal'")
   parser.add_argument('-query_ratio', type=float,
        help='ratio between number of steps the agent is assisted and total number of steps (tau)')
   parser.add_argument('-n_subgoal_steps', type=int,
        help='number of next actions suggested by a subgoal')
   parser.add_argument('-subgoal_vocab', type=str,
        help='subgoal vocabulary')

   # Help-requesting teacher hyperparameters
   parser.add_argument('--deviate_threshold', type=float)
   parser.add_argument('--uncertain_threshold', type=float)
   parser.add_argument('--visit_threshold', type=int)

   # Don't touch these ones
   parser.add_argument('-backprop_softmax', type=int, default=1)
   parser.add_argument('-backprop_ask_features', type=int)

   # Budget Features
   parser.add_argument('-max_ask_budget', type=int, default=20,
        help='budget upperbound')

   # Evaluation
   parser.add_argument('-success_radius', type=float,
        help='success radius')
   parser.add_argument('-eval_only', type=int,
        help='evaluation mode')
   parser.add_argument('-multi_seed_eval', type=int,
        help='evaluate with multiple seeds (automatically set -eval_only 1)')
   parser.add_argument('-teacher_interpret', type=int,
        help='0 = evaluate with indirect advisor    1 = evaluate with direct advisor')

   # Others
   parser.add_argument('-device_id', type=int,
        help='gpu id')
   parser.add_argument('-no_room', type=int,
        help='train or evaluate with the no_room dataset (when using this, set -data_dir noroom)')

   parser.add_argument('-double_request_budget_every', type=int,
        help='(for training) cirriculum training by doubling request budget once in a while')

   parser.add_argument('-bc_epochs', type=int,
        help='Number of iterations to apply behavior cloning')
   parser.add_argument('-bcui_epochs', type=int,
        help='Number of iterations to apply behavior cloning under intervention')
   parser.add_argument('-bcui_prob', type=float,
        help='Probability of performing bcui')

   parser.add_argument('-start_point_radius', type=int,
        help='Start point radius')

   parser.add_argument('-random_agent', type=int, default=0,
        help='Agent that randomly selects navigation actions')

   parser.add_argument('-forward_agent', type=int, default=0,
        help='Agent that always selects action 1')

   parser.add_argument('-shortest_agent', type=int, default=0,
        help='Optimal shortest-path agent')

   parser.add_argument('-no_ask', type=int, default=0,
        help='No ask')

   parser.add_argument('-alpha', type=float,
        help='Weight between positive and negative navigation losses')

   parser.add_argument('-debug', type=int, default=0)

   parser.add_argument('-ask_baseline', type=str, default=None,
        help='Help-request teacher baseline')
   parser.add_argument('-ask_every_epochs', type=int)

   parser.add_argument('-instruction_baseline', type=str,
        help='Instruction type baseline')

   parser.add_argument('-perfect_interpretation', type=int,
        help='provide perfect assistance interpretation')

   """
   parser.add_argument('-random_ask', type=str, default=None,
        help='Random ask')
   parser.add_argument('-ask_every', type=str, default=None,
        help='Ask every K steps')
   """

   return parser
