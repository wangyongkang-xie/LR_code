from abc import ABC
from abc import abstractmethod
import os
import numpy as np
import torch
from aggretator import Baseline


def random_str(n):
    return hex(int.from_bytes(os.urandom(n), byteorder='big'))[2:]


class ModelBase(ABC):
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])

    @abstractmethod
    def update_grads(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass


class PytorchModel(ModelBase):
    def __init__(self, torch, model_class, device, init_model_path: str = '', lr: float = 0.01,
                 optim_name: str = 'Adam', cuda: bool = True, **kwargs):
        """Pytorch 封装.

        参数：
            torch: torch 库
            model_class: 训练模型类
            init_model_path: 初始模型路径
            lr: 学习率
            optim_name: 优化器类名称
            cuda: 是否需要使用cuda
        """
        super().__init__(**kwargs)
        self.device = device
        self.torch = torch
        self.model_class = model_class
        self.init_model_path = init_model_path
        self.lr = lr
        self.optim_name = optim_name
        self.cuda = cuda

        self._init_params()

    def _init_params(self):
        self.model = self.model_class()
        if self.init_model_path:
            self.model.load_state_dict(self.torch.load(self.init_model_path))

        if self.cuda and self.torch.cuda.is_available():
            self.model = self.model.to(self.device)

        self.optimizer = getattr(self.torch.optim, self.optim_name)(self.model.parameters(), lr=self.lr)

    def update_grads(self, grads):
        self.optimizer.zero_grad()

        for k, v in self.model.named_parameters():
            v.grad = grads[k].type(v.dtype)

        self.optimizer.step()

    def update_params(self, params):

        for k, v in self.model.named_parameters():
            v.data = params[k]

        return self.model

    def update_params_foolsgold(self, params):

        for i, (k, v) in enumerate(self.model.named_parameters()):
            v.data = params[i]

        return self.model

    def load_model(self, path, force_reload=False):
        if force_reload is False and self.load_from_path == path:
            return

        self.load_from_path = path
        self.model.load_static_dict(self.torch.load(path))

    def save_model(self, path):
        base = os.path.dirname(path)
        if not os.path.exists(base):
            os.makedirs(base)

        self.torch.save(self.model.state_dict(), path)

        return path


class BaseBackend(ABC):
    @abstractmethod
    def mean(self, data):
        data = np.array(data)

        return data.mean(axis=0)


class NumpyBackend(BaseBackend):
    def mean(self, data):
        return super().mean(data=data)


class PytorchBackend(BaseBackend):
    def __init__(self, torch, cuda=True):
        self.torch = torch
        if cuda:
            if self.torch.cuda.is_available():
                self.cuda = True
        else:
            self.cuda = False

    def mean(self, data, dim=0):
        return self.torch.tensor(
            data,
            device=self.torch.cuda.current_device() if self.cuda else torch.device("cpu"),
        ).mean(dim=dim)

    def sum(self, data, dim=0):
        return self.torch.tensor(
            data,
            device=self.torch.cuda.current_device() if self.cuda else torch.device("cpu"),
        ).sum(dim=dim)

    def _check_model(self, model):
        if not isinstance(model, PytorchModel):
            raise ValueError(
                "model must be type of PytorchModel not {}".format(
                    type(model)))

    def update_grads(self, model, grads):
        self._check_model(model=model)
        return model.update_grads(grads=grads)

    def update_params_1(self, model, params):
        self._check_model(model=model)
        return model.update_params(params=params)

    def update_params_2(self, model, params):
        self._check_model(model=model)
        return model.update_params_foolsgold(params=params)

    def load_model(self, model, path, force_reload=False):
        self._check_model(model=model)
        return model.load_model(path=path, force_reload=force_reload)

    def save_model(self, model, path):
        self._check_model(model=model)
        return model.save_model(path)


class Aggregator(object):
    def __init__(self, model, backend):
        self.model = model
        self.backend = backend


class FederatedAveragingGrads(Aggregator):
    def __init__(self, model, framework=None):
        self.framework = framework or getattr(model, 'framework')

        if framework is None or framework == 'numpy':
            backend = NumpyBackend
        elif framework == 'pytorch':
            backend = PytorchBackend(torch=torch)
        else:
            raise ValueError(
                'Framework {} is not supported!'.format(framework))

        super().__init__(model, backend)
        self.baselines = Baseline()

    def aggregate_grads(self, grads, agg_method, beta, del_num, partion):
        """Aggregate model gradients to models.

        Args:
            data: a list of grads' information
                item format:
                    {
                        'n_samples': xxx,
                        'named_grads': xxx,
                    }
        """
        global selected_weights
        if agg_method == "TrimmedMean":
            selected_weights = self.baselines.cal_TrimmedMean(grads, beta=beta)
        elif agg_method == "Krum":
            selected_weights = self.baselines.cal_Krum(grads, num_byz=del_num)
        elif agg_method == "GeoMed":
            selected_weights = self.baselines.cal_GeoMed(grads)
        elif agg_method == "Theroy":
            selected_weights = self.baselines.cal_theroy(grads, num=partion, del_num=del_num, device=self.model.device)
        elif agg_method == "Normal":
            selected_weights = self.baselines.cal_Normal(grads, device=self.model.device)
        elif agg_method == "NoDetect":
            selected_weights = self.baselines.cal_NoDetect(grads, device=self.model.device)
        elif agg_method == "foolsgold":
            selected_weights = self.baselines.cal_foolsgold(grads)
        elif agg_method == "rfa":
            selected_weights = self.baselines.call_rfa(grads, device=self.model.device)
        else:
            print("####### Invalid Benchmark Option #######")
        if (agg_method == "foolsgold") | (agg_method == "rfa"):
            return self.backend.update_params_2(self.model, selected_weights)
        return self.backend.update_params_1(self.model, selected_weights)
        # return self.backend.update_params(self.model, params=aggregate_grads(grads=grads,backend=self.backend))

    def save_model(self, path):
        return self.backend.save_model(self.model, path=path)

    def load_model(self, path, force_reload=False):
        return self.backend.load_model(self.model,
                                       path=path,
                                       force_reload=force_reload)

    def __call__(self, grads, agg_method, beta, del_num, partion):
        """Aggregate grads.

        Args:
            grads -> list: grads is a list of either the actual grad info
            or the absolute file path  of grad info.
        """
        if not grads:
            return

        if not isinstance(grads, list):
            raise ValueError('grads should be a list, not {}'.format(
                type(grads)))
        # actual_grads = grads
        # actual_grads = sample(grads,8)
        return self.aggregate_grads(grads, agg_method, beta, del_num, partion)
