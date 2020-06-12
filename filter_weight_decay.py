def group_weight(model):
    group_decay = []
    group_no_decay = []
    for name, param in model.named_parameters():
        # if ('conv' and 'weight') in name:  # = conv or weight in name why?????????
        if 'conv' in name and 'weight' in name:
            group_decay.append(param)
            # print('decay:', name)
        else:
            group_no_decay.append(param)
            # print('no_decay:', name)

    assert len(list(model.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups
