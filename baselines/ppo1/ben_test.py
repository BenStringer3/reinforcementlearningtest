# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 09:29:25 2018

@author: ben
"""

#!/usr/bin/env python3

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
import gym, logging
from baselines import logger
from baselines.ppo1.pposgd_simple import pposgd as solution
from baselines.common.atari_wrappers import make_atari, wrap_deepmind

def train(env_id, num_timesteps, seed, kind):
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()

    #make a place to save tensorboard stuff
    dir = '/tmp/mylogdir/'
    #if this is the first process
    if rank == 0:
        from fs.osfs import OSFS
        import os
        # not sure what this does -ben
        logger.configure()
        try:
            folder = OSFS(dir)
        except:
            os.makedirs(dir)
            folder = OSFS(dir)
        test_n = len(list(n for n in folder.listdir('./') if n.startswith('test')))
        this_test = dir + "test" + str(test_n + 1)
        for i in range(1, MPI.COMM_WORLD.Get_size()):
            MPI.COMM_WORLD.send(test_n+1, dest=i, tag=11)
    else: #if this is not the first process
        #not sure what this does -ben
        logger.configure(format_strs=[])
        #wait unti process (rank) 0 has made a folder for the rest of the processes to write to
        test_n = MPI.COMM_WORLD.recv(source=0, tag=11)
        this_test = dir + "test" + str(test_n)

    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)#make_atari(env_id)
    # env = gym.normalize(gym.GymEnv(env_id))
    def policy_fn(name, ob_space, ac_space): # pylint: disable=W0613
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space, kind=kind)
    env = bench.Monitor(env, logger.get_dir() and
        osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    # env = wrap_deepmind(env)
    env.seed(workerseed)
    sol = solution(rank, this_test)
    sol.learn( env, policy_fn,
        max_timesteps=int(num_timesteps * 1.1),
        timesteps_per_actorbatch=128,#256,
        clip_param=0.2, entcoeff=0.01,
        optim_epochs=3, optim_stepsize=2.5e-4, optim_batchsize=32,
        gamma=0.99, lam=0.95,
        schedule='linear'
    )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='PongNoFrameskip-v4')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(10e6))
    parser.add_argument('--kind', help='type of network (small, large, dense)', default='large')
    args = parser.parse_args()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed, kind=args.kind)

if __name__ == '__main__':
    main()
