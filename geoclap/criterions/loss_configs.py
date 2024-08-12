from yacs.config import CfgNode as CN
losscfg = CN()

# Loss parameters initialized from: https://github.com/naver-ai/pcmepp/tree/main/configs

def get_loss_config(name='pcmepp', pseudo_match_alpha=0.1, vib_beta=0.0001):
    
    if name == 'pcmepp':
        losscfg.name = 'pcmepp'
        losscfg.init_negative_scale = 5
        losscfg.init_shift = 5
        losscfg.vib_beta = vib_beta
        losscfg.prob_distance = 'csd'
        losscfg.smoothness_alpha =  pseudo_match_alpha
    
    elif name == 'pcme':
        losscfg.name = 'pcme'
        losscfg.init_negative_scale = 5
        losscfg.init_shift = 5
        losscfg.num_samples = 8
        losscfg.vib_beta = vib_beta
        losscfg.prob_distance = 'non_squared_l2'
        losscfg.pdist_fn = 'iterative'
    
    elif name == 'infonce':
        losscfg.name = 'info_nce'
        losscfg.init_tau = 1
    
    else:
        raise NotImplementedError("{} loss not implemented.\
                                  Only supported loss types are: infonce, pcme, pcmepp.",name)
    return dict(losscfg)

        
