"""Interfaces for ClientModel and ServerModel."""

from abc import ABC, abstractmethod
import numpy as np
import random
# import tensorflow as tf

# from baseline_constants import OptimLoggingKeys, AGGR_MEAN, AGGR_GEO_MED
#
# from utils.model_utils import batch_data
# from utils.tf_utils import graph_size
import torch
import torch.nn.functional as F

class ServerModel(object):
    # def __init__(self, model):
    #     self.model = model
    #     self.rng = model.rng

    @property
    def size(self):
        return self.model.size

    @property
    def cur_model(self):
        return self.model

    # def send_to(self, clients):
    #     """Copies server model variables to each of the given clients
    #
    #     Args:
    #         clients: list of Client objects
    #     """
    #     var_vals = {}
    #     with self.model.graph.as_default():
    #         all_vars = tf.trainable_variables()
    #         for v in all_vars:
    #             val = self.model.sess.run(v)
    #             var_vals[v.name] = val
    #     for c in clients:
    #         with c.model.graph.as_default():
    #             all_vars = tf.trainable_variables()
    #             for v in all_vars:
    #                 v.load(var_vals[v.name], c.model.sess)

    @staticmethod
    def weighted_average_oracle(points, weights):
        """Computes weighted average of atoms with specified weights

        Args:
            points: list, whose weighted average we wish to calculate
                Each element is a list_of_np.ndarray
            weights: list of weights of the same length as atoms
        """
        tot_weights = torch.sum(weights)
        weighted_updates = [torch.zeros_like(v) for v in points[0]]

        for w, p in zip(weights, points):
            for j, weighted_val in enumerate(weighted_updates):
                weighted_val += (w / tot_weights) * p[j]

        return weighted_updates

    def update(self, updates, device, aggregation="median", maxiter=4):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
            aggregation: Algorithm used for aggregation. Allowed values are:
                [ 'mean', 'geom_median']
            max_update_norm: Reject updates larger than this norm,
            maxiter: maximum number of calls to the Weiszfeld algorithm if using the geometric median
            @rtype: object
        """
        # def accept_update(u):
        #     norm = np.linalg.norm([np.linalg.norm(x) for x in u[1]])
        #     return not (np.isinf(norm) or np.isnan(norm))
        # all_updates = updates
        # updates = [u for u in updates if accept_update(u)]
        # if len(updates) < len(all_updates):
        #     print('Rejected {} individual updates because of NaN or Inf'.format(len(all_updates) - len(updates)))
        # if len(updates) == 0:
        #     print('All individual updates rejected. Continuing without update')
        #     return 1, False
        points = [list(item['named_grads'].values()) for item in updates]
        # points = torch.FloatTensor(points)
        # points = points.to(device)
        # points = [u[1] for u in updates]
        alphas = torch.FloatTensor([item['n_samples'] for item in updates]).to(device)
        if aggregation == "mean":
            weighted_updates = self.weighted_average_oracle(points, alphas)
            num_comm_rounds = 1
        elif aggregation == "median":
            weighted_updates, num_comm_rounds, _ = self.geometric_median_update(points, alphas,device, maxiter=maxiter)
        else:
            raise ValueError('Unknown aggregation strategy: {}'.format(aggregation))

        # update_norm = np.linalg.norm([np.linalg.norm(v) for v in weighted_updates])
        #
        # if max_update_norm is None or update_norm < max_update_norm:
        #     with self.model.graph.as_default():
        #         all_vars = tf.trainable_variables()
        #         for i, v in enumerate(all_vars):
        #             init_val = self.model.sess.run(v)
        #             v.load(np.add(init_val, weighted_updates[i]), self.model.sess)
        #     updated = True
        # else:
        #     print('\t\t\tUpdate norm = {} is too large. Update rejected'.format(update_norm))
        #     updated = False

        return weighted_updates

    def save(self, path=None):
        return self.model.saver.save(self.model.sess, path) if path is not None else None

    def load(self, path):
        return self.model.saver.restore(self.model.sess, path)

    def close(self):
        self.model.close()

    @staticmethod
    def geometric_median_update(points, alphas, device, maxiter=4, eps=1e-5, verbose=False, ftol=1e-6):
        """Computes geometric median of atoms with weights alphas using Weiszfeld's Algorithm
        """
        # alphas = np.asarray(alphas, dtype=points[0][0].dtype) / sum(alphas)
        alphas = alphas / torch.sum(alphas)
        median = ServerModel.weighted_average_oracle(points, alphas)
        num_oracle_calls = 1

        # logging
        obj_val = ServerModel.geometric_median_objective(median, points, alphas,device)
        logs = []
        log_entry = [0, obj_val, 0, 0]
        logs.append(log_entry)
        if verbose:
            print('Starting Weiszfeld algorithm')
            print(log_entry)

        # start
        for i in range(maxiter):
            prev_median, prev_obj_val = median, obj_val
            weights = torch.FloatTensor([alpha / max(eps, ServerModel.l2dist(median, p, device)) for alpha, p in zip(alphas, points)])
            weights = weights / torch.sum(weights)
            median = ServerModel.weighted_average_oracle(points, weights)
            num_oracle_calls += 1
            obj_val = ServerModel.geometric_median_objective(median, points, alphas,device)
            log_entry = [i+1, obj_val,
                         (prev_obj_val - obj_val)/obj_val,
                         ServerModel.l2dist(median, prev_median, device)]
            logs.append(log_entry)
            if verbose:
                print(log_entry)
            if abs(prev_obj_val - obj_val) < ftol * obj_val:
                break
        return median, num_oracle_calls, logs

    @staticmethod
    def l2dist(p1, p2,device):
        """L2 distance between p1, p2, each of which is a list of nd-arrays"""
        # dis = [np.linalg.norm(x1.cpu().numpy() - x2.cpu().numpy()) for x1, x2 in zip(p1, p2)]
        # print(dis)
        # distance = np.linalg.norm(dis)
        # print(distance)
        # return np.linalg.norm([np.linalg.norm(x1 - x2) for x1, x2 in zip(p1, p2)])
        # p1 = p1.view(-1,1)
        # p2 = p2.view(-1, 1)
        dis_ = [F.pairwise_distance(x1.view(1,-1),x2.view(1,-1)).cpu().numpy() for x1, x2 in zip(p1, p2)]
        dis_ = torch.tensor(dis_).to(device)
        dis_ = dis_.float()
        distance_ = F.pairwise_distance(dis_.view(1,-1),torch.zeros_like(dis_).view(1,-1))
        # print(distance_)
        return distance_

    @staticmethod
    def geometric_median_objective(median, points, alphas,device):
        """Compute geometric median objective."""
        return sum([alpha * ServerModel.l2dist(median, p,device) for alpha, p in zip(alphas, points)])
