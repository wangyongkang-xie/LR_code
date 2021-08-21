from collections import Counter

import torch
import torch.nn.functional as F
# import preprocess
import torch.optim as optim
# from learning_model import LeNet
# import numpy as np
from torch import nn
from attacks import sign_flipping_attack, random_attack, backward


def user_round_train(train_loader, model_new,
                     device, attackmode,
                     model_old, local_epoch,
                     batch_size, log, backdoor,
                     helper):
    # for X,Y in data:
    #     Data = CompDataset(X=X, Y=Y)
    # print(Counter(Y))
    # print(np.any(np.isnan(X)))
    # Data = CompDataset(X=train_loader[0], Y=train_loader[1])
    # global my_param, total_loss, correct
    # data_iterator = torch.utils.data.DataLoader(
    #     Data,
    #     batch_size=batch_size,  # 原来为320
    #     shuffle=True,
    # )
    global correct, total_loss, my_param, count
    if (attackmode == 5) & (backdoor):
        AGENT_POISON_AT_THIS_ROUND = True
    else:
        AGENT_POISON_AT_THIS_ROUND = False
    opti = optim.Adam(model_new.parameters(), lr=0.1)
    model_new.train()
    _, data_iterator = train_loader
    for i in range(local_epoch):
        correct = 0
        total_loss = 0
        count = 0
        poison_data_count = 0
        for batch_idx, batch in enumerate(data_iterator):
            if AGENT_POISON_AT_THIS_ROUND:
                data, target, poison_num = helper.get_poison_batch(batch, adversarial_index=-1, evaluation=False)
                poison_data_count += poison_num
            else:
                data, target = helper.get_batch(data_iterator, batch, evaluation=False)

            data = data.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            data, target = data.to(device), target.to(device)
            count += len(target)
            opti.zero_grad()
            output = model_new(data)
            loss = F.cross_entropy(output, target.long())
            total_loss += loss
            loss.backward()
            nn.utils.clip_grad_norm_(model_new.parameters(), max_norm=10, norm_type=2)
            opti.step()
            pred = output.argmax(
                dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.long().view_as(pred)).sum().item()

    model_old_dic = {}
    for name, param in model_old.named_parameters():
        model_old_dic[name] = param.detach()

    grads = {'n_samples': count, 'named_grads': {}}
    if AGENT_POISON_AT_THIS_ROUND:
        for name, param in model_new.named_parameters():
            # param = param.detach
            param = param.detach()
            if attackmode == 1:  # same value
                # my_param = same_value_attack(param)
                pass
            elif attackmode == 2:  # sign flipping
                my_param = sign_flipping_attack(param)
            elif attackmode == 3:  # random value
                my_param = random_attack(param,device)
            elif attackmode == 4:  # under development
                # print(param.size(),model_old_dic[name].size())
                my_param = backward(param, model_old_dic[name])
            # elif attackmode == 5:  # wrong label
            #     pass
            else:
                my_param = param
            grads['named_grads'][name] = helper.params["scale_factor"] * my_param
    else:
        for name, param in model_new.named_parameters():
            # param = param.detach
            param = param.detach()
            if attackmode == 1:  # same value
                # my_param = same_value_attack(param)
                pass
            elif attackmode == 2:  # sign flipping
                my_param = sign_flipping_attack(param)
            elif attackmode == 3:  # random value
                my_param = random_attack(param,device)
            elif attackmode == 4:  # under development
                # print(param.size(),model_old_dic[name].size())
                my_param = backward(param, model_old_dic[name])
            # elif attackmode == 5:  # wrong label
            #     pass
            else:
                my_param = param
            grads['named_grads'][name] = my_param
        # grads['named_grads'][name] = param.detach().cpu().numpy()#转换为参数
    if True:
        # log.info('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
        #     total_loss, 100. * correct / len(data_iterator.dataset)))
        log.info('Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
            total_loss, 100. * correct / count))

    return grads
# if __name__ == "__main__":
#     userrounddata = preprocess.UserRoundData()
#
#     # data_x,data_y = userrounddata.round_data(1, 0, 1000,1)
#     # print(data_y)
#     # count = Counter(data_y)
#     #     print(count)
#     # print(data)
#     X,Y = userrounddata.round_data(0,0,n_round_samples=-1,sign=-1)
#     print(np.any(np.isnan(X)))
#     model = FLModel()
#     device = torch.device("cuda")
#     for i in range(100):
#         user_round_train(X, Y, model, device, debug=False)
