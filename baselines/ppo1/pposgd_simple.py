from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import math

MODEL_NAME = 'model4'
REWARD_SCALE = 10

class pposgd():

    def __init__(self, rank, this_test):
        self.max_rew = -math.inf
        self.min_loss = math.inf
        self.rank = rank
        self.this_test = this_test
        sess = tf.get_default_session()
        self.writer =  tf.summary.FileWriter(self.this_test + '/rank_' + str(self.rank), sess.graph)  # ben

    def traj_segment_generator(self, pi, env, horizon, stochastic):
        t = 0
        ac = env.action_space.sample() # not used, just so we have the datatype
        new = True # marks if we're on first timestep of an episode
        ob = env.reset()

        cur_ep_ret = 0 # return in current episode
        cur_ep_len = 0 # len of current episode
        ep_num = 0
        ep_rets = [] # returns of completed episodes in this segment
        ep_lens = [] # lengths of ...

        # Initialize history arrays
        obs = np.array([ob for _ in range(horizon)])
        rews = np.zeros(horizon, 'float32')
        vpreds = np.zeros(horizon, 'float32')
        news = np.zeros(horizon, 'int32')
        acs = np.array([ac for _ in range(horizon)])
        prevacs = acs.copy()
        # writer = tf.summary.FileWriter('/tmp/mylogdir/pi/test26/')

        while True:
            prevac = ac

            ac, vpred = pi.act(stochastic, ob)
            summary = tf.Summary(value=[tf.Summary.Value(tag="vpred", simple_value=vpred)])
            self.writer.add_summary(summary, t)
            summary = tf.Summary(value=[tf.Summary.Value(tag="top_or_front", simple_value=ac[2])])
            self.writer.add_summary(summary, t)
            summary = tf.Summary(value=[tf.Summary.Value(tag="ac_bool", simple_value=ac[1])])
            self.writer.add_summary(summary, t)
            summary = tf.Summary(value=[tf.Summary.Value(tag="rmba_sel", simple_value=ac[0])])
            self.writer.add_summary(summary, t)
            # go_bool = (np.tanh(ac[3]) + 1) / 2
            # ac_bool = (np.tanh(ac[4]) + 1) / 2
            # top_or_front = (np.tanh(ac[5]) + 1) / 2
            # # heading_targeted=(np.tanh(ac[2]) + 1) / 2 * math.pi*2
            # summary = tf.Summary(value=[tf.Summary.Value(tag="top_or_front", simple_value=np.round(top_or_front))])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="ac_bool", simple_value=np.round(ac_bool))])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="pos_x_tar", simple_value=(np.tanh(ac[0]) + 1) / 2 * (20.0) -0.0)])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="pos_y_tar", simple_value=(np.tanh(ac[1]) + 1) / 2 * (20.0) -0.0)])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="vel_x_tar", simple_value=(np.tanh(ac[2]) + 1) / 2 * (0.33*2) -0.33)])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="vel_y_tar", simple_value=(np.tanh(ac[3]) + 1) / 2 * (0.33*2) -0.33)])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="heading_targeted", simple_value=np.round(heading_targeted))])
            # self.writer.add_summary(summary, t)
            # summary = tf.Summary(value=[tf.Summary.Value(tag="go_bool", simple_value=np.round(go_bool))])
            # self.writer.add_summary(summary, t)
            # Slight weirdness here because we need value function at time T
            # before returning segment [0, T-1] so we get the correct
            # terminal value
            if t > 0 and t % horizon == 0:
                yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                        "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                        "ep_rets" : ep_rets, "ep_lens" : ep_lens}
                # Be careful!!! if you change the downstream algorithm to aggregate
                # several of these batches, then be sure to do a deepcopy
                ep_rets = []
                ep_lens = []
            i = t % horizon
            obs[i] = ob
            vpreds[i] = vpred
            news[i] = new
            acs[i] = ac
            prevacs[i] = prevac

            ob, rew, new, dist_dict = env.step(ac)

            rew = rew*REWARD_SCALE #ben
            summary = tf.Summary(value=[tf.Summary.Value(tag="min_dist_all", simple_value=dist_dict["min_dist_all"])])
            self.writer.add_summary(summary, t)
            summary = tf.Summary(value=[tf.Summary.Value(tag="min_dist_ac", simple_value=dist_dict["min_dist_ac"])])
            self.writer.add_summary(summary, t)

            if self.rank == 0:
                env.render()
            rews[i] = rew

            cur_ep_ret += rew
            cur_ep_len += 1
            if new:
                ep_num += 1
                summary = tf.Summary(value=[tf.Summary.Value(tag="rew", simple_value=cur_ep_ret)])
                self.writer.add_summary(summary, ep_num)
                self.writer.flush()
                if cur_ep_ret > self.max_rew:
                    U.save_state('/tmp/models/' + MODEL_NAME + '_best.ckpt')
                    self.max_rew = cur_ep_ret

                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0
                ob = env.reset()
            t += 1

    def add_vtarg_and_adv(self, seg, gamma, lam):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
        vpred = np.append(seg["vpred"], seg["nextvpred"])
        T = len(seg["rew"])
        seg["adv"] = gaelam = np.empty(T, 'float32')
        rew = seg["rew"]
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1-new[t+1]
            delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        seg["tdlamret"] = seg["adv"] + seg["vpred"]

    def learn(self, env, policy_func, *,
            timesteps_per_actorbatch, # timesteps per actor per update
            clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
            optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
            gamma, lam, # advantage estimation
            max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
            callback=None, # you can do anything in the callback, since it takes locals(), globals()
            adam_epsilon=1e-5,
            schedule='constant' # annealing for stepsize parameters (epsilon and adam)
            ):
        # Setup losses and stuff
        # ----------------------------------------
        ob_space = env.observation_space
        ac_space = env.action_space
        print("Object space:", ob_space, "Action_space", ac_space)
        pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
        oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
        atarg = tf.placeholder(dtype=tf.float32, shape=[None], name="atarg") # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None], name="ret") # Empirical return

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        clip_param = clip_param * lrmult # Annealed cliping parameter epislon

        ob = U.get_placeholder_cached(name="ob")
        ac = pi.pdtype.sample_placeholder([None])

        with tf.name_scope("kloldnew"):
            kloldnew = oldpi.pd.kl(pi.pd)
        with tf.name_scope("entropy"):
            ent = pi.pd.entropy()
        with tf.name_scope("meankl"):
            meankl = U.mean(kloldnew)
        with tf.name_scope("mean_entropy"):
            meanent = U.mean(ent)
        with tf.name_scope("pol_entpen"):
            pol_entpen = (-entcoeff) * meanent

        with tf.name_scope("optimization"):
            with tf.name_scope("ratio"):
                ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
            with tf.name_scope("surr1"):
                surr1 = ratio * atarg # surrogate from conservative policy iteration
            with tf.name_scope("surr2"):
                surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
            with tf.name_scope("pol_surr"):
                pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
            with tf.name_scope("vf_loss"):
                vf_loss= U.mean(tf.square(pi.vpred - ret))
            with tf.name_scope("total_loss"):
                total_loss = 10e3*pol_surr + pol_entpen + vf_loss
            losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent, total_loss]
            loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent", "total"]

        # totes_loss = tf.placeholder(dtype=tf.float32, shape=[], name="totes_loss")
        var_list = pi.get_trainable_variables()
        # my_flatGrad = U.flatgrad(totes_loss, var_list)
        # justComputeGrad = U.function([ob, ac, atarg, ret, lrmult], [U.flatgrad(total_loss, var_list)])
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list, nan_override=True, writer=self.writer)])
        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

        U.initialize()
        adam.sync()

        loss_summary = tf.summary.scalar("total_loss", total_loss)

        merged = tf.summary.merge_all()  # ben
        tf.global_variables_initializer().run()  # ben

        #save graph for visualization
        if self.rank == 0:
            graph_def = tf.get_default_graph().as_graph_def()
            graphpb_txt = str(graph_def)

            with open('/tmp/mylogdir/graphpb.txt', 'w') as f:
                f.write(graphpb_txt)
        if self.rank == 0:
            try:
                U.load_state('/tmp/models/' + MODEL_NAME + '.ckpt')
            except:
                # os.mkdir('/tmp/models')
                U.save_state('/tmp/models/' + MODEL_NAME + '.ckpt')
                print('no state loaded, saving newly built one for next time')
            for i in range(1, MPI.COMM_WORLD.Get_size()):
                MPI.COMM_WORLD.send(True, dest=i, tag=12)
        else:
            MPI.COMM_WORLD.recv(source=0, tag=12)
            U.load_state('/tmp/models/' + MODEL_NAME + '.ckpt')
    # Prepare for rollouts
        # ----------------------------------------
        seg_gen = self.traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

        while True:
            if callback: callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************"%iters_so_far)


            seg = seg_gen.__next__()
            # summary = tf.Summary(value=[tf.Summary.Value(tag="rew", simple_value=seg["rew"][-1])])
            # writer.add_summary(summary, iters_so_far)
            self.add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            losses = []
            #tewwsting to see if nan is in losses
            # totes_loss = tf.placeholder(dtype=tf.float32, shape=[None])
            # var_list = pi.get_trainable_variables()
            # my_flatGrad = U.flatgrad(totes_loss, var_list)
            # justComputeGrad = U.function([totes_loss], [U.flatgrad(total_loss, var_list)])
            # lossandgrad = U.function([ob, ac, atarg, ret, lrmult],
            #                          losses + [U.flatgrad(total_loss, var_list, nan_override=True, writer=self.writer)])
            # for batch in d.iterate_once(optim_batchsize):
            #     newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)

            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    #    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))
            if iters_so_far % 25 == 0:
                U.save_state('/tmp/models/' + MODEL_NAME + '.ckpt')
            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
                # if losses[-1] < self.min_loss:
                #     U.save_state('/tmp/models/' + MODEL_NAME + '.ckpt')
                #     self.min_loss = losses[-1]
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_"+name, lossval)
                summary = tf.Summary(value=[tf.Summary.Value(tag="loss_"+name, simple_value=lossval)])
                self.writer.add_summary(summary, iters_so_far)
            logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
            lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
            lens, rews = map(self.flatten_lists, zip(*listoflrpairs))
            lenbuffer.extend(lens)
            rewbuffer.extend(rews)
            logger.record_tabular("EpLenMean", np.mean(lenbuffer))
            logger.record_tabular("EpRewMean", np.mean(rewbuffer))
            logger.record_tabular("EpThisIter", len(lens))
            episodes_so_far += len(lens)
            timesteps_so_far += sum(lens)
            iters_so_far += 1
            logger.record_tabular("EpisodesSoFar", episodes_so_far)
            logger.record_tabular("TimestepsSoFar", timesteps_so_far)
            logger.record_tabular("TimeElapsed", time.time() - tstart)

            summary = tf.Summary(value=[tf.Summary.Value(tag="cur_lrmult ", simple_value=cur_lrmult )])
            self.writer.add_summary(summary, iters_so_far)


            if MPI.COMM_WORLD.Get_rank()==0:
                logger.dump_tabular()

    def flatten_lists(self, listoflists):
        return [el for list_ in listoflists for el in list_]
