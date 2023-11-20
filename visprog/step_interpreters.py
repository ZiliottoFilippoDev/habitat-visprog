import cv2
import os
import torch
# import openai
import functools
import numpy as np
import io, tokenize
from PIL import Image,ImageDraw,ImageFont,ImageFilter

from .blablabla import PointnavModel

# from .nms import nms
# from vis_utils import html_embed_image, html_colored_span, vis_masks

translate = { 0: 'STOP',
              1: 'MOVE FORWARD',
              2: 'TURN LEFT',
              3: 'TURN LEFT'
            }

def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result

# 
# parse_results = {
#     'step_name' : 'NAV',
#     'observations': 'obs',
#     'output_var': 'NAV0'
# }


class PointNavInterpreter():
    step_name = 'NAV'

    def __init__(self, agent):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = agent
        self.model.eval()

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        obs_batch = parse_result['args']['observation']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return obs_batch, output_var
    
    def predict_action(self, obs_batch):

        action_data = self.model(
            obs_batch,
            test_recurrent_hidden_states,
            prev_actions,
            **args,
        )

        action_step = [a.item() for a in action_data.env_actions.cpu()]
        return action_step

    def execute(self,prog_step,inspect=False):
        obs_batch, output_var = self.parse(prog_step)
        img = prog_step.state[obs_batch['rgb']]
        
        action = self.predict_action(obs_batch)

        # save action into the word (e.g. action = tensor([3]) --> "MOVEFORWARD")
        prog_step.state[output_var] = translate[action]
        prog_step.state[output_var + '_RGB_IMAGE'] = img

        return action

    
def register_step_interpreters(dataset='pointnav'):
    if dataset=='pointnav':
        return dict(
            NAV=PointNavInterpreter(),
        )