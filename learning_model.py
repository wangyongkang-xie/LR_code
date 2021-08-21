import torch.nn.functional as F
from torch import nn
# from baselines import Baseline,Baseline_single
# class GlobalModel(object):
#     """docstring for GlobalModel"""
#     def __init__(self, output_file="stats.txt"):
#         self.model = self.build_model()
#         self.current_weights = self.model.get_weights()
#         # print(len(self.model.layers))
#         # for convergence check
#         self.prev_train_loss = None
#
#         self.best_loss = None
#         self.best_weight = None
#         self.best_round = -1
#
#         # all rounds; losses[i] = [round#, timestamp, loss]
#         # round# could be None if not applicable
#         self.train_losses = []
#         self.valid_losses = []
#         self.train_accuracies = []
#         self.valid_accuracies = []
#         self.pre_train_losses = []
#         self.pre_train_accuracies = []
#
#         self.training_start_time = int(round(time.time()))
#
#         self.baselines = Baseline()
#         self.baselines_single = Baseline_single()
#
#         self.output_file = output_file
#
#     def build_model(self):
#         raise NotImplementedError()
#
#     # client_updates = [(w, n)..]
#     def update_weights(self, client_weights, client_sizes):
#         new_weights = [np.zeros(w.shape) for w in self.current_weights]
#         total_size = np.sum(client_sizes)
#
#         for c in range(len(client_weights)):
#             for i in range(len(new_weights)):
#                 new_weights[i] += client_weights[c][i] * client_sizes[c] / total_size
#         self.current_weights = new_weights
#
#     def update_weights_baseline(self, client_weights, client_sizes, agg_method):
#         # ["normal_atten", "atten", "TrimmedMean", "Krum", "GeoMed"]
#         global selected_weights
#         if agg_method == "TrimmedMean":
#             selected_weights = self.baselines.cal_TrimmedMean(client_weights)
#         elif agg_method == "Krum":
#             selected_weights = self.baselines.cal_Krum(client_weights)
#         elif agg_method == "GeoMed":
#             selected_weights = self.baselines.cal_GeoMed(client_weights)
#         elif agg_method == "theroy":
#             selected_weights = self.baselines.cal_theroy(client_weights,5)
#         else:
#             print("####### Invalid Benchmark Option #######")
#
#         # with open('server_trimmedMean.log', 'a') as fw:
#         #     for item in client_weights:
#         #         fw.write( "{} [INFO] Client weights: {}".format(datetime.datetime.now(), np.array2string(np.array(item[-1]))) )
#         #     fw.write( '{} [Selected]: {}\n'.format(datetime.datetime.now(), selected_weights[-1]) )
#         #     fw.write( '------------------------\n')
#
#         self.current_weights = selected_weights
#
#     def update_weights_with_attention(self, client_weights, client_sizes, attention, attack_label):
#         new_weights = [np.zeros(w.shape) for w in self.current_weights]
#         total_size = np.sum(client_sizes)
#         attention = np.asarray(attention)
#         # print("new attention", attention)
#         client_sizes = np.asarray(client_sizes) / total_size
#         print("client_sizes", client_sizes)
#         scores = np.multiply(attention, client_sizes)
#         # print("scores", scores)
#         scores_norm = scores / np.sum(scores)
#         print("scores_norm", scores_norm)
#         # exit()
#
#         for c in range(len(client_weights)):
#             for i in range(len(new_weights)):
#                 new_weights[i] += (client_weights[c][i]) * scores_norm[c]
#
#         with open('server_attention_sign_flipping.log', 'a') as fw:
#             for item in client_weights:
#                 fw.write( "{} [INFO] Client weights: {}".format(datetime.datetime.now(), np.array2string(np.array(item[-1]))) )
#             fw.write( '{} [Weights]: {}\n'.format(datetime.datetime.now(), attention) )
#             fw.write( '{} [Attack_Label]: {}\n'.format(datetime.datetime.now(), " ".join(attack_label)) )
#             fw.write( '\n------------------------\n')
#         self.current_weights = new_weights
#
#     def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
#         total_size = np.sum(client_sizes)
#         aggr_loss = np.sum(client_losses[i] / total_size * client_sizes[i]
#                 for i in range(len(client_sizes)))
#         aggr_accuraries = np.sum(client_accuracies[i] / total_size * client_sizes[i]
#                 for i in range(len(client_sizes)))
#         return aggr_loss, aggr_accuraries
#
#     def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
#         cur_time = int(round(time.time())) - self.training_start_time
#         aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
#         self.train_losses += [[cur_round, cur_time, aggr_loss]]
#         self.train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
#         with open(self.output_file, 'w') as outfile:
#             json.dump(self.get_stats(), outfile)
#         return aggr_loss, aggr_accuraries
#
#         # cur_round coule be None
#     def aggregate_pre_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
#         cur_time = int(round(time.time())) - self.training_start_time
#         aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
#         self.pre_train_losses += [[cur_round, cur_time, aggr_loss]]
#         self.pre_train_accuracies += [[cur_round, cur_time, aggr_accuraries]]
#         with open(self.output_file, 'w') as outfile:
#             json.dump(self.get_stats(), outfile)
#         return aggr_loss, aggr_accuraries
#
#     def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, cur_round):
#         cur_time = int(round(time.time())) - self.training_start_time
#         aggr_loss, aggr_accuraries = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
#         self.valid_losses += [[cur_round, cur_time, aggr_loss]]
#         self.valid_accuracies += [[cur_round, cur_time, aggr_accuraries]]
#         with open(self.output_file, 'w') as outfile:
#             json.dump(self.get_stats(), outfile)
#         return aggr_loss, aggr_accuraries
#
#     def get_stats(self):
#         return {
#             "train_loss": self.train_losses,
#             "valid_loss": self.valid_losses,
#             "train_accuracy": self.train_accuracies,
#             "valid_accuracy": self.valid_accuracies,
#             "pre_train_loss": self.pre_train_losses,
#             "pre_train_accuracy": self.pre_train_accuracies,
#         }
#
# class ResidualBlock(nn.Module):
#     def __init__(self, inchannel, outchannel, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.left = nn.Sequential(
#             nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.shortcut = nn.Sequential()
#         if stride != 1 or inchannel != outchannel:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(outchannel)
#             )
#
#     def forward(self, x):
#         out = self.left(x)
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out
#
#
# class ResNet(nn.Module):
#     def __init__(self, ResidualBlock, num_classes=10):
#         super(ResNet, self).__init__()
#         self.inchannel = 64
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#         )
#         self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
#         self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
#         self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
#         self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
#         self.fc = nn.Linear(512, num_classes)
#
#     def make_layer(self, block, channels, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
#         layers = []
#         for stride in strides:
#             layers.append(block(self.inchannel, channels, stride))
#             self.inchannel = channels
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out
# def ResNet18():
#     return ResNet(ResidualBlock)
#
#
# class GlobalModel_MNIST_CNN(GlobalModel):
#     def __init__(self):
#         super(GlobalModel_MNIST_CNN, self).__init__()
#
#     def build_model(self):
#         # ~5MB worth of parameters
#         model = Sequential()
#         model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
#         model.add(Conv2D(64, (3, 3), activation='relu'))
#    # keras.layers.Conv2D(filters, kernel_size, strides=(1, 1)
#    # filters: the dimensionality of the output space
#
#         model.add(MaxPooling2D(pool_size=(2, 2)))
#         model.add(Dropout(0.25))
#         model.add(Flatten())
#         model.add(Dense(128, activation='relu'))
#         model.add(Dropout(0.5))
#         model.add(Dense(10, activation='softmax'))
#         model.compile(loss=keras.losses.categorical_crossentropy,
#                       optimizer=keras.optimizers.Adadelta(),
#                       metrics=['accuracy'])
#         return model
# class LeNet(nn.Module):
#     def __init__(self):
#         super(LeNet, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3, 1, 2), nn.ReLU(),
#                                    nn.MaxPool2d(2, 2))
#
#         self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(),
#                                    nn.MaxPool2d(2, 2))
#
#         self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120),
#                                  nn.BatchNorm1d(120), nn.ReLU())
#
#         self.fc2 = nn.Sequential(
#             nn.Linear(120, 84),
#             nn.BatchNorm1d(84),
#             nn.ReLU(),
#             nn.Linear(84, 10))
#             # 最后的结果一定要变为 10，因为数字的选项是 0 ~ 9
#
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x.view(x.size()[0], -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
#
#
# class RESNET18_MODEL(GlobalModel):
#     def __init__(self):
#         super(RESNET18_MODEL, self).__init__()
#
#     def build_model(self, channel = 1, width = 28, height = 28, nbr_classes = 62):
#         # ~5MB worth of parameters
#         model = resnet.ResnetBuilder.build_resnet_18( (channel, width, height), nbr_classes )
#
#         model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
#
#         print(model.summary())
#         return model
#
# class LENET5_MODEL(GlobalModel):
#     def __init__(self):
#         super(LENET5_MODEL, self).__init__()
#     def build_model(self):
#         model = Sequential()
#
#         model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(28,28,1)))
#         model.add(MaxPooling2D(pool_size = 2, strides = 2))
#
#         model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
#         model.add(MaxPooling2D(pool_size = 2, strides = 2))
#
#         model.add(Flatten())
#
#         model.add(Dense(units=120, activation='relu'))
#
#         model.add(Dense(units=84, activation='relu'))
#
#         model.add(Dense(units=10, activation = 'softmax'))
#
#         model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
#
#         print(model.summary())
#
#         return model

class LENET5_MODEL_FEMNIST(nn.Module):
    def __init__(self):
        super(LENET5_MODEL_FEMNIST, self).__init__()
        # def build_model(self):
        #     super(LeNet, self).__init__()
        #     def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3,
                      # stride=2,
                      padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d((2, 2))
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(32, 64, 3, 2, 1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(64, 64, 2, 2, 0),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU()
        # )
        self.mlp1 = nn.Linear(7 * 7 * 32, 1024)
        self.mlp2 = nn.Linear(1024, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        x = F.log_softmax(x, dim=1)
        # output = F.softmax(x, dim=0)
        return x
        # model = Sequential()
        #
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding = "same", activation='relu', input_shape=(28,28,1)))
        # model.add(MaxPooling2D(pool_size = 2, strides = 2))
        #
        # model.add(Conv2D(filters=32, kernel_size=(3, 3), padding = "same", activation='relu'))
        # model.add(MaxPooling2D(pool_size = 2, strides = 2))
        #
        # model.add(Flatten())
        #
        # model.add(Dense(units=1024, activation='relu'))
        #
        # model.add(Dense(units=62, activation = 'softmax'))
        #
        # sgd = optimizers.SGD(lr=0.06)
        # model.compile(optimizer = sgd, loss = 'categorical_crossentropy', metrics = ['accuracy'])
        #
        # print(model.summary())


class LENET5_MODEL_MNIST(nn.Module):
    def __init__(self):
        super(LENET5_MODEL_MNIST, self).__init__()
        # def build_model(self):
        #     super(LeNet, self).__init__()
        #     def __init__(self):
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=3, ), nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU())
        self.pool = nn.MaxPool2d((2, 2))
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.mlp1 = nn.Linear(24 * 24 * 64, 128)
        self.mlp2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        # x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.dropout2(x)
        x = self.mlp2(x)
        x = F.log_softmax(x, dim=1)
        # output = F.softmax(x, dim=0)
        return x


# class STACKED_LSTM(GlobalModel):
#     def __init__(self, char_embedding_size, lstm_hidden_size=256):
#         super(STACKED_LSTM, self).__init__()
#
#         self.char_embedding_size = char_embedding_size
#         self.lstm_hidden_size = lstm_hidden_size
#
#     def build_model(self):
#
#         input = Input(shape=(None, self.char_embedding_size))
#         #- We define the model as variable-length (even though all training data has fixed length).
#         # This allows us to generate longer sequences during inference.
#
#         h_1 = LSTM(self.lstm_hidden_size, return_sequences=True)(input)
#         h_2 = LSTM(self.lstm_hidden_size, return_sequences=True)(h_1)
#
#         #  Apply a single dense layer to all timesteps of the resulting sequence to convert back to characters
#         out = TimeDistributed(Dense(self.char_embedding_size, activation='softmax'))(h_2)
#
#         model = Model(input, out)
#
#         opt = keras.optimizers.Adam()
#
#         model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#         #- For each timestep the model outputs a probability distribution over all characters. Categorical crossentopy mean
#         #  that we try to optimize the log-probability of the probability of the correct character (averaged over all
#         #  characters in all sequences.
#
#         print("LSTM Model Summary\n",model.summary())
#
#         return model
class lr_model(nn.Module):
    def __init__(self):
        super(lr_model, self).__init__()
        self.linear = nn.Linear(784, 62, bias=True)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.linear(x)
        # x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    # m = LENET5_MODEL()
    # m.build_model()
    # r = RESNET18_MODEL()
    # r.build_model()
    fe = lr_model()
    print(fe)
