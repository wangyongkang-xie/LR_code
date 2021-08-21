import argparse
import json
import logging
import random
from datetime import datetime
import os
import numpy as np
import yaml
import torch
from torchvision.transforms import transforms
import test

import csv_record
from context import FederatedAveragingGrads
from context import PytorchModel
from learning_model import LENET5_MODEL_FEMNIST, lr_model
# from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from train import user_round_train
import shutil
import config
import image_helper
import warnings

logger = logging.getLogger("logger")
warnings.filterwarnings("ignore")


# print(torch.cuda.is_available())

class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir, model_choice, device):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}
        self.current_round_grads = []
        self.model_choice = model_choice
        self.init_model_path = init_model_path
        self.device = device
        self.aggr = FederatedAveragingGrads(
            model=PytorchModel(torch=torch,
                               model_class=self.model_choice,
                               device=self.device,
                               init_model_path=self.init_model_path,
                               optim_name='Adam'),
            framework='pytorch',
        )

        self.testworkdir = testworkdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads):
        self.current_round_grads.append(grads)

    def aggregate(self, agg_method, beta, del_num, partion):
        self.aggr(self.current_round_grads, agg_method, beta, del_num, partion)
        # self.aggr.aggregate_grads(self.current_round_grads)
        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.rounds_model_path[self.round] = path
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        self.round += 1
        self.current_round_grads = []

        return info


# class FedAveragingGradsTestSuit(unittest.TestCase):
class FedAveragingGradsTestSuit():
    # RESULT_DIR = 'result'
    # N_VALIDATION = 50000
    TEST_BASE_DIR = './tmp'

    def __init__(self):
        # self.aggregation = 'GeoMed'
        self.seed = 0
        self.use_cuda = True
        self.batch_size = 64
        self.test_batch_size = 1000
        self.lr = 0.01
        self.n_max_rounds = 150
        self.log_interval = 10
        self.n_round_samples = 1600
        self.log = './log'
        self.testbase = self.TEST_BASE_DIR
        self.model_choice = lr_model
        self.args_mk()
        # self.result = './result'
        self.result = './result' + '/' + self.args.aggregation + '/' + self.args.data + '/' + str(self.args.attack_mode) \
                        + '/' + str(self.args.attack_ratio)
        # self.log()
        with open(f'./{self.args.params}', 'r') as f:
            self.params_loaded = yaml.load(f)
        if not os.path.exists(self.log):
            os.mkdir(self.log)
        if not os.path.exists(self.result):
            os.makedirs(self.result)
        logger.addHandler(logging.FileHandler(
            filename=f'{self.log}/log' + str(self.args.attack_mode) + '_' + str(self.args.attack_ratio) + '.txt'))
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)
        # self.logger = logging.getLogger("loggerfile")
        self.testworkdir = os.path.join(self.testbase, self.args.aggregation,
                                        str(self.args.attack_mode), str(self.args.attack_ratio))
        """下次运行时删除原来的文件夹，再创建"""
        if os.path.exists(self.testworkdir):
            shutil.rmtree(self.testworkdir)

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        torch.manual_seed(self.seed)

        if not os.path.exists(self.init_model_path):
            torch.save(self.model_choice().state_dict(), self.init_model_path)

        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir, model_choice=self.model_choice, device=self.args.device)

        # if not os.path.exists(self.RESULT_DIR):
        #     os.makedirs(self.RESULT_DIR)

    """
    def femnist_process(self):
        random.seed(5)
        np.random.seed(5)
        test_file = "D:/博士/信息论实验/femnist/test"
        train_leaf_dir = "D:/博士/信息论实验/femnist/wyk/train"
        # test_leaf_dir = "D:\博士\code/femnist/wyk/test"
        test_data_x, test_y = self.read_data(test_file)
        self.writers = os.listdir(train_leaf_dir)
        self.test_data = CompDataset(test_data_x, test_y)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1000)
        self.n_users = len(self.writers)
    """

    def read_data(self, test_data_dir):
        '''parses data in given train and test data directories

        assumes:
        - the data in the input directories are .json files with
            keys 'users' and 'user_data'
        - the set of train set users is the same as the set of test set users

        Return:
            clients: list of client ids
            groups: list of group ids; empty list if none found
            train_data: dictionary of train data
            test_data: dictionary of test data
        '''
        test_data_x = []
        test_data_y = []
        test_files = os.listdir(test_data_dir)
        test_files = [f for f in test_files if f.endswith('.json')]
        for f in test_files:
            file_path = os.path.join(test_data_dir, f)
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
            for k, v in cdata['user_data'].items():
                test_data_x.extend(v['x'])
                test_data_y.extend(v['y'])
        test_data_x = np.asarray(test_data_x)
        test_data_y = np.asarray(test_data_y)
        # train_data_x = np.reshape(train_data_x, (train_data_x.shape[0],
        #                                                28, 28,1))
        test_data_x = np.reshape(test_data_x, (test_data_x.shape[0], 1,
                                               28, 28))
        return test_data_x, test_data_y

    """
    def mnist_process(self):
        random.seed(5)
        np.random.seed(5)
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train_left, x_train_right, y_train_left, y_train_right = train_test_split(x_train, y_train, test_size=0.5)
        Minst_1 = Mnist(20, 10, x_train_left, y_train_left)
        Minst_2 = Mnist(20, 10, x_train_right, y_train_right)
        self.writers = Minst_1.client()  # noniid数据
        writers_right = Minst_2.client()
        self.writers.extend(writers_right)
        self.data_test = Minst_1.test_data()
        # self.x_test, self.y_test = data_test[0], data_test[1]
        self.test_data = CompDataset(self.data_test[0], self.data_test[1])
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=1000)
        self.n_users = len(self.writers)
    """
    def mnist_process_100(self):
        random.seed(5)
        np.random.seed(5)
        current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
        if self.params_loaded['type'] == config.TYPE_MNIST:
            self.helper = image_helper.ImageHelper(current_time=current_time, params=self.params_loaded,
                                                   name=self.params_loaded.get('name', 'mnist'))
            self.helper.load_data()
        else:
            self.helper = None
        self.writers = self.helper.train_data
        self.test_loader = self.helper.test_data
        self.n_users = len(self.writers)
        # print(self.n_users)
        # for i in range(self.n_users):
        #     _, data1 = self.writers[i]
        #     print(len(data1.dataset))

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def args_mk(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--params', default='mnist_params.yaml')
        parser.add_argument('--device', default='cuda:0')
        parser.add_argument("--data", type=str, default='mnist')
        parser.add_argument("--data_process", type=str, default='mnist_100')
        parser.add_argument('--aggregation', type=str,
                            choices=["NoDetect", "Normal", "Theroy", "TrimmedMean", "Krum",
                                     "GeoMed", "foolsgold", "rfa"],
                            default="NoDetect")
        parser.add_argument("--batch_size", type=int, default=64)
        parser.add_argument("--local_epoch", type=int, default=10)
        parser.add_argument("--select_ratio", type=float, default=0.04)  # 选取20个client
        parser.add_argument("--attack_mode", type=int, default=5)
        parser.add_argument("--backdoor", type=bool, default=True)
        parser.add_argument("--attack_ratio", type=float, default=0.25)
        parser.add_argument("--beta", type=float, default=0.25)
        parser.add_argument("--del_num", type=int, default=10)
        parser.add_argument("--partion", type=int, default=5)
        parser.add_argument("--distributed", type=str, default=False)
        self.args = parser.parse_args()

    def test_federated_averaging(self):
        random.seed(5)
        np.random.seed(5)
        # self.mnist_process()
        global data
        self.aggregation = self.args.aggregation
        # self.
        if self.args.data_process == 'femnist':
            pass
            # self.femnist_process()
        elif self.args.data_process == 'mnist':
            pass
            # self.mnist_process()
        else:
            self.mnist_process_100()
        logger.info("##### Arguements #####")
        logger.info(self.args)
        logger.info("##########")
        NUM_CLIENT = int(self.n_users * self.args.select_ratio)
        # torch.manual_seed(self.seed)
        # device = torch.device("cuda" if self.use_cuda else "cpu")
        device = torch.device("cuda:0")
        training_start = datetime.now()
        model = None
        model_old = self.model_choice()
        for r in range(1, self.n_max_rounds + 1):
            if self.args.distributed:
                if r in self.params_loaded["distributed_attack"]:
                    self.args.backdoor = True
                else:
                    self.args.backdoor = False
            else:
                if r in self.params_loaded["central_attack"]:
                    self.args.backdoor = True
                else:
                    self.args.backdoor = False
            # clients = []
            # selected_writers = np.random.choice(self.n_users, size=NUM_CLIENT, replace=False)
            selected_writers = [i for i in range(self.params_loaded["number_of_total_participants"])]
            attack_mode = [0] * len(selected_writers)
            attack_number = int(self.args.attack_ratio * len(selected_writers))
            if not self.args.distributed:
                # print("可进入")
                for attack_idx in range(attack_number):
                    attack_mode[attack_idx] = self.args.attack_mode
            else:
                if r in self.params_loaded["distributed_attack"]:
                    attack_mode[r-8] = self.args.attack_mode
            logger.info("Selected Writers", selected_writers, len(selected_writers))
            logger.info("Attack Modes", attack_mode)
            path = self.ps.get_latest_model()
            model_old.load_state_dict(torch.load(path))
            model_old.to(device)
            start = datetime.now()
            for u in range(0, len(selected_writers)):
                model = self.model_choice()
                model.load_state_dict(torch.load(path))
                model = model.to(device)
                to_pil_image = transforms.ToPILImage()
                # x, y = self.writers[u]
                if self.args.data_process == 'femnist':
                    # femnist = FEMNIST(self.writers[selected_writers[u]])
                    # data = femnist.fake_non_iid_data()
                    pass
                if self.args.data_process == 'mnist_100':  # backdoor attack
                    data = self.writers[selected_writers[u]]
                else:
                    data = self.writers[selected_writers[u]]
                # print(data[1])
                grads = user_round_train(data, model, device, attack_mode[u], model_old,
                                         self.args.local_epoch,
                                         self.args.batch_size,
                                         logger,
                                         self.args.backdoor,
                                         self.helper)
                # grads_ = deepcopy(grads)
                self.ps.receive_grads_info(grads=grads)
                # self.ps.aggr.
            self.ps.aggregate(self.aggregation, self.args.beta, self.args.del_num, self.args.partion)
            logger.info('\nRound {} cost: {}, total training cost: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
            ))
            # print(model)
            # if model is not None:
            #     self.predict(model,
            #                  device,
            #                  self.train_loader,
            #                  prefix="Train")
            #     self.save_testdata_prediction(model=model, device=device)
            # for k,v in model.named_parameters():
            #     print(v.mean())
            if model is not None:
                self.predict(model,
                             device,
                             self.test_loader,
                             prefix="Test")
            # self.save_testdata_prediction(model=model, device=device)

    def save_prediction(self, loss, acc, type):
        # if isinstance(predition, (np.ndarray, )):
        #     predition = predition.reshape(-1).tolist()
        # with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
        # fout.writelines(os.linesep.join([str(n) for n in predition]))

        if 'total' == type:
            loss_file = self.result + '/' + 'total_test_loss_结果存放.txt'
            acc_file = self.result + '/'  + 'total_test_acc_结果存放.txt'
        elif 'backdoor' == type:
            loss_file = self.result + '/' + 'backdoor_test_loss_结果存放.txt'
            acc_file = self.result + '/' + 'backdoor_test_acc_结果存放.txt'
        else:
            loss_file = self.result + '/' + 'nobackdoor_test_loss_结果存放.txt'
            acc_file = self.result + '/' + 'nobackdoor_test_acc_结果存放.txt'
        with open(loss_file, 'a') as file_handle:
            file_handle.write(str(loss))
            file_handle.write('\n')
        with open(acc_file, 'a') as file_handle:
            file_handle.write(str(acc))
            file_handle.write('\n')

    def predict(self, model, device, test_loader, prefix=""):
        path = self.ps.get_latest_model()
        model.load_state_dict(torch.load(path))
        # for k,v in model.named_parameters():
        #     print(v.mean())
        model.eval()
        model = model.to(device)
        epoch_loss, epoch_acc, epoch_corret, epoch_total = test.clean_test(helper=self.helper,
                                                                           model=model,
                                                                           is_poison=False,
                                                                           visualize=True,
                                                                           agent_name_key="global")

        self.save_prediction(epoch_loss, epoch_acc, 'nobackdoor')

        epoch_loss, epoch_acc_p, epoch_corret, epoch_total = test.adv_test(helper=self.helper,
                                                                           model=model,
                                                                           is_poison=True,
                                                                           visualize=True,
                                                                           agent_name_key="global")
        self.save_prediction(epoch_loss, epoch_acc, 'backdoor')

        #     logger.info(
        #         '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #             prefix, test_loss, correct_back + correct_noback, len(test_loader.dataset), acc), )
        #
        #     logger.info(
        #         '{} set: Test backdoor average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #             prefix, test_loss_back, correct_back, data_len_back, acc_back), )
        #
        #     logger.info(
        #         '{} set: Test nobackdoor average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #             prefix, test_loss_noback, correct_noback, len(test_loader.dataset) - data_len_back, acc_noback), )
        # else:
        #     test_loss /= len(test_loader.dataset)
        #     acc = 100. * correct / len(test_loader.dataset)
        #     self.save_prediction(test_loss, acc)
        #     # print(classification_report(real, prediction))
        #     logger.info(
        #         '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        #             prefix, test_loss, correct, len(test_loader.dataset), acc), )

"""
    def make_print_to_file(self, path=None):
        '''
        path， it is a path for save your log about fuction print
        example:
        use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
        :return:
        '''
        import sys
        import os
        # import config_file as cfg_file
        import sys
        import datetime

        class Logger(object):
            def __init__(self, filename="Default.log", path="./"):
                self.terminal = sys.stdout
                self.log = open(os.path.join(path, filename), "a", encoding='utf8', )

            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)

            def flush(self):
                pass

        fileName = datetime.datetime.now().strftime('day' + '%Y_%m_%d') + self.args.aggregation + '_femnist_' + str(
            self.args.attack_mode) + '_' + str(self.args.attack_ratio)
        sys.stdout = Logger(fileName + '.log', path=path)

        #############################################################
        # 这里输出之后的所有的输出的print 内容即将写入日志
        #############################################################
        print(fileName.center(60, '*'))
        
"""


def main():
    # make_print_to_file(path='./')
    Fed = FedAveragingGradsTestSuit()
    # Fed.make_print_to_file(Fed.testworkdir)
    Fed.test_federated_averaging()


if __name__ == '__main__':
    main()
