#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 10:35:38 2018

@author: matthewszhang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 12:23:38 2018

@author: matthewszhang
"""

import copy

def intermediate_goal(state_dict):
    s = copy.deepcopy(state_dict)
    
    gpr_1 = state_dict['gpr_1'][0]
    gpr_2 = state_dict['gpr_2'][0]
    
    for i in range(len(state_dict['state']) - 1):
        if state_dict['state'][i] > state_dict['state'][i+1]:
            # if first is greater than second
            if state_dict['gpr_1'][0] != state_dict['state'][i]:
                if state_dict['ptr'][0] > i:
                    s['ptr'][0] -= 1
                    return s
                elif state_dict['ptr'][0] < i:
                    s['ptr'][0] += 1
                    return s
                else:
                    s['gpr_1'][0] = state_dict['state'][i]
                    return s
            else:
                if state_dict['ptr'][0] > i+1:
                    s['ptr'][0] -= 1
                    return s
                elif state_dict['ptr'][0] < i+1:
                    s['ptr'][0] += 1
                    return s
                elif state_dict['gpr_2'][0] != state_dict['state'][i+1]:
                    s['gpr_2'][0] = state_dict['state'][i+1]
                    return s
                else:
                    s['state'][i+1] = state_dict['gpr_1'][0]
                    return s
                    
        elif state_dict['state'][i] == state_dict['state'][i+1]:
            # assume first has been swapped with second
            if ((state_dict['gpr_2'][0] < state_dict['state'][i]) and (state_dict['gpr_2'][0] not in state_dict['state'])) or \
                ((state_dict['gpr_1'][0] < state_dict['state'][i]) and (state_dict['gpr_1'][0] not in state_dict['state'])):
                if state_dict['ptr'][0] > i:
                    s['ptr'][0] -= 1
                    return s
                elif state_dict['ptr'][0] < i:
                    s['ptr'][0] += 1
                    return s
                elif state_dict['gpr_2'][0] < state_dict['state'][i]:
                    s['state'][i] = state_dict['gpr_2'][0]
                    return s
                else:
                    s['state'][i] = state_dict['gpr_1'][0]
                    return s
            elif ((state_dict['gpr_2'][0] > state_dict['state'][i+1]) and (state_dict['gpr_2'][0] not in state_dict['state'])) or \
                ((state_dict['gpr_1'][0] > state_dict['state'][i+1]) and (state_dict['gpr_1'][0] not in state_dict['state'])):
                if state_dict['ptr'][0] > i+1:
                    s['ptr'][0] -= 1
                    return s
                elif state_dict['ptr'][0] < i+1:
                    s['ptr'][0] += 1
                    return s
                elif state_dict['gpr_2'][0] > state_dict['state'][i+1]:
                    s['state'][i+1] = state_dict['gpr_2'][0]
                    return s
                else:
                    s['state'][i+1] = state_dict['gpr_1'][0]
                    return s
    
    s['comp_flag'][0] = 1
    return s        