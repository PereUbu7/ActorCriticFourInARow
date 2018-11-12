import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from time import time

import os.path
from fourInARowWrapper import FourInARowWrapper

if "../" not in sys.path:
  sys.path.append("../")
from lib import plotting


matplotlib.style.use('ggplot')

env = FourInARowWrapper(1)

def invertBoard(inBoard):
    invertedBoard = np.array(inBoard)

    board_shape = inBoard.shape

    #print("Shape:", board_shape)

    for x in range(board_shape[0]):
        for y in range(board_shape[1]):
            invertedBoard[x][y][0] = inBoard[x][y][1]
            invertedBoard[x][y][1] = inBoard[x][y][0]

    return invertedBoard

class ConvolutionalNetwork():
    def __init__(self, scope="conv_net"):
        with tf.variable_scope(scope):
            self.board = tf.placeholder(tf.float32, (None, 7, 6, 2), "board")
            #self.player = tf.placeholder(tf.float32, (None, 2), "player")

            self.filter1 = tf.Variable(tf.random_normal([3, 3, 2, 20]), name="filter1")
            self.filter2 = tf.Variable(tf.random_normal([2, 2, 20, 40]), name="filter2")

            self.board_norm = tf.nn.batch_normalization(x=self.board, mean=0, variance=1, offset=1, scale=1, variance_epsilon=1e-7)

            self.conv1 = tf.nn.conv2d(
                input=self.board_norm,
                filter=self.filter1,
                strides=(1, 1, 1, 1),
                padding="SAME"
            )

            self.deconv1 = tf.nn.conv2d_transpose(self.conv1, self.filter1, tf.shape(self.board_norm), strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", name="deconv1")

            self.l1 = tf.nn.leaky_relu(self.conv1, 0.1)

            # self.l1p = tf.layers.max_pooling2d(
            #     inputs = self.l1,
            #     pool_size = (3, 3),
            #     strides = (2, 2),
            #     padding = "VALID",
            #     data_format='channels_last',
            #     name="maxPooling"
            # )

            self.conv2 = tf.nn.conv2d(
                input=self.l1,
                filter=self.filter2,
                strides=(1, 1, 1, 1),
                padding="SAME"
            )

            self.deconv2 = tf.nn.conv2d_transpose(self.conv2, self.filter2, tf.shape(self.l1), strides=(1, 1, 1, 1), padding="SAME",
                                                  data_format="NHWC", name="deconv2")
            self.deconv2_1 = tf.nn.conv2d_transpose(self.deconv2, self.filter1, tf.shape(self.board_norm), strides=(1, 1, 1, 1), padding="SAME", data_format="NHWC", name="deconv2_1")

            self.outLayerConv = tf.nn.leaky_relu(self.conv2, 0.1)

            self.board_flat = tf.reshape(self.board_norm, [tf.shape(self.board_norm)[0], 84])
            self.outLayerConv_flat = tf.reshape(self.outLayerConv, [tf.shape(self.outLayerConv)[0], 1680])

            self.board_and_out = tf.concat([self.board_flat, self.outLayerConv_flat], 1)

            self.outLayer_pre = tf.contrib.layers.fully_connected(
                inputs=self.board_and_out,
                num_outputs=500,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="outLayer"
            )

            self.outLayer = tf.contrib.layers.dropout(
                self.outLayer_pre,
                keep_prob=0.5,
            )


class Trainer():
    def __init__(self, learning_rate=0.001, convNet=None, policy=None, policyLossFactor=0.1, value=None, valueLossFactor=0.1, scope="trainer"):
        with tf.variable_scope(scope):
            self.policy = policy
            self.value = value
            self.convNet = convNet
            self.loss = policyLossFactor * policy.loss + valueLossFactor * value.loss

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def update(self, board, td_target, td_error, action, avaliableColumns, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.policy.target: td_error, self.policy.action: action,
                    self.policy.validColumnsFilter: avaliableColumns,
                     self.value.target: td_target, self.convNet.board: board}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    def evalFilters(self, board, sess=None):
        sess = sess or tf.get_default_session()

        board_exp = np.expand_dims(board, axis=0)

        feed_dict = {self.convNet.board: board_exp}
        layer1, layer2 = sess.run([self.convNet.deconv1, self.convNet.deconv2_1], feed_dict)

        return layer1, layer2


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, entropyFactor=0.1, shared_layers=None, scope="policy_estimator"):
        with tf.variable_scope(scope):
            self.shared_layers = shared_layers
            if shared_layers is not None:
                self.board = shared_layers.board
                self.input = shared_layers.outLayer
                #self.player = shared_layers.player
            else:
                print("Needs shared_layers parameter")
                return -1

            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.validColumnsFilter = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="validColumnsFilter")

            self.l1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=100,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="l2"
            )

            self.l1_dropout = tf.contrib.layers.dropout(
                self.l1,
                keep_prob=0.9,
            )

            self.mu = tf.contrib.layers.fully_connected(
                inputs=self.l1_dropout,
                num_outputs=env.action_space.high-env.action_space.low,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="mu")

            self.mu = tf.squeeze(self.mu)

            self.mu = tf.multiply(self.mu, self.validColumnsFilter) + 1e-6

            self.mu = tf.divide(self.mu, tf.reduce_sum(self.mu))

            self.dist = tf.contrib.distributions.Categorical(probs=self.mu, dtype=tf.float32)

            # Draw sample
            self.action = self.dist.sample()

            # Loss and train op
            self.loss = -self.dist.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration
            self.loss -= entropyFactor * self.dist.entropy()


    def predict(self, env, sess=None):
        sess = sess or tf.get_default_session()

        player = np.expand_dims(env.state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(env.state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(env.state[1]), axis=0)

        action, mu = sess.run([self.action, self.mu], {self.shared_layers.board: board, self.validColumnsFilter: np.expand_dims(env.getAvaliableColumns(), axis=0)})
        return action, mu


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, shared_layers=None, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.shared_layers = shared_layers
            if shared_layers is not None:
                self.board = shared_layers.board
                self.input = shared_layers.outLayer
            else:
                print("Needs shared_layers parameter")
                return -1

            #self.player = tf.placeholder(tf.float32, (None, 2), "player")
            self.target = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="target")

            self.l1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=100,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="l2"
            )

            self.l1_dropout = tf.contrib.layers.dropout(
                self.l1,
                keep_prob=0.9,
            )

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.l1_dropout,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="output_layer")

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)


    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        player = np.expand_dims(state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(state[1]), axis=0)
        #state = featurize_state(state)
        return sess.run(self.value_estimate, {self.shared_layers.board: board})


def actor_critic(env, estimator_policy, estimator_value, trainer, num_episodes, discount_factor=1.0, player2=True, positiveRewardFactor=1.0, negativeRewardFactor=1.0, batch_size=1):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy: Policy Function to be optimized
        estimator_value: Value function approximator, used as a critic
        trainer: our training class
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        player2: True if computer plays player2, False if user does
        positiveRewardFactor: Factor bla bla bla reward
        negativeRewardFactor: Factor bla bla bla
        batch_size: Batch size

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_td_error=np.zeros(num_episodes),
        episode_value_loss=np.zeros(num_episodes),
        episode_policy_loss=np.zeros(num_episodes),
        episode_kl_divergence=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    batch_board = np.zeros((batch_size, 7, 6, 2))
    batch_player = np.zeros((batch_size, 2))
    batch_td_target = np.zeros((batch_size, 1))
    batch_td_error =np.zeros((batch_size, 1))
    batch_action =np.zeros((batch_size, 1))
    batch_avaliableColumns = np.zeros((batch_size, 7))

    batch_pos = 0

    game = 1

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset(i_episode % 2 + 1)

        episode = []

        probas = None
        last_turn = False
        done = False
        last_state = None
        action = None
        reward = None

        if game % 5000 == 10:
            player2 = True
        elif game % 5000 == 0:
            player2 = False

        if game == num_episodes-3:
            player2 = False

        # One step in the environment
        for t in itertools.count():
            # Save avaliable columns
            if not done:
                avaliableColumns = env.getAvaliableColumns()

            currentPlayerBeforeStep = env.getCurrentPlayer()

            action_tmp = action

            # Take a step
            if currentPlayerBeforeStep == 1 or currentPlayerBeforeStep == 2 and player2 and not done:
                action, probas = estimator_policy.predict(env)
                action = action[0]
                probas = probas[0]
            elif not done:
                try:
                    action = int(input("Give a column number: ")) - 1
                except ValueError:
                    print("Wrong input! Setting action to 1")
                    action = 0
                probas = None

            if not done:
                next_state, reward, step_done, _ = env.step(action)

                if game % 1000 == 0:
                    layer1, layer2 = trainer.evalFilters(next_state[1])
                    plotting.plotNNFilter(next_state[1], layer1, layer2)
                    #plotting.plotNNFilter(layer2)


                if step_done:
                    pass

                if t > 0:
                    state_tmp = last_state
                    last_state = state
                    reward_tmp = -reward*negativeRewardFactor
                else:
                    state_tmp = state
                    last_state = state
                    reward_tmp = -reward*negativeRewardFactor


            elif done and not last_turn:
                state_tmp = episode[-2].state
                reward_tmp = reward*positiveRewardFactor
            else:
                break




            if t > 0:
                episode.append(Transition(
                    state=state_tmp, action=action_tmp, reward=reward_tmp, next_state=next_state, done=done))

                player = None
                if episode[-1].state[0][0] == 1:
                    player = "X"
                elif episode[-1].state[0][1] == 1:
                    player = "O"
                # Update statistics
                stats.episode_rewards[i_episode] += episode[-1].reward
                stats.episode_lengths[i_episode] = t

                # Calculate TD Target
                value_next = estimator_value.predict(episode[-1].next_state, )

                td_target = episode[-1].reward + discount_factor * value_next


                td_error = td_target - estimator_value.predict(episode[-1].state)


                stats.episode_td_error[i_episode] += td_error

                batch_player[batch_pos] = episode[-1].state[0]
                # Network always plays as player one
                if batch_player[batch_pos][0] == 1:
                    batch_board[batch_pos] = episode[-1].state[1]
                else:
                    batch_board[batch_pos] = invertBoard(episode[-1].state[1])

                batch_td_target[batch_pos] = td_target
                batch_td_error[batch_pos] = td_error
                batch_action[batch_pos] = episode[-1].action
                batch_avaliableColumns[batch_pos] = avaliableColumns

                batch_pos += 1

                if batch_pos == batch_size:
                    # Update both networks
                    loss = trainer.update(batch_board, batch_td_target, batch_td_error, batch_action, batch_avaliableColumns)
                    loss = loss[0][0]
                    batch_pos = 0

                    print("Updates network. Loss:", loss)



                    stats.episode_value_loss[i_episode] += loss

                    if probas is not None and last_probas is not None:
                        kl_div = 0
                        for i in range(probas.size):
                            kl_div += probas[i]*np.log(probas[i]/last_probas[i])
                        stats.episode_kl_divergence[i_episode] += kl_div

                # Print out which step we're on, useful for debugging.
                print("Player {}: Action {}, Reward {}, TD Error {}, at Step {} @ Game {} @ Episode {}/{} ({})\n".format(
                        player , episode[-1].action+1, episode[-1].reward, td_error, t, game, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if game == num_episodes or env.getCurrentPlayer() == 2 and not player2:
                env.render()
                if probas is not None:
                    out = " "
                    for i in range(probas.size):
                        out += "%.1f " % probas[i]
                    print(out)

            last_probas = probas

            if done:
                last_turn = True
                game += 1


            if step_done:
                done = True

            state = next_state

    return stats

tf.reset_default_graph()

start = time()

batch_size = 500

global_step = tf.Variable(0, name="global_step", trainable=False)
conv_net = ConvolutionalNetwork()
policy_estimator = PolicyEstimator(entropyFactor=1e-0, shared_layers=conv_net)
value_estimator = ValueEstimator(shared_layers=conv_net)
trainer = Trainer(learning_rate=1e-3, convNet=conv_net, policy=policy_estimator, policyLossFactor=1, value=value_estimator, valueLossFactor=1e-0)

variables = tf.contrib.slim.get_variables_to_restore()
variables_to_restore = [v for v in variables if v.name.split('/')[0]!='trainer' and v.name.split('/')[0]!='policy_estimator' and v.name.split('/')[0]!='value_estimator']
variables_to_init = [v for v in variables if v.name.split('/')[0]!='conv_net']

for v in variables_to_restore:
    print(v)

saver = tf.train.Saver(variables_to_restore)

with tf.Session() as sess:
    try:
        saver.restore(sess, "tmp/model5.ckpt")
        sess.run(tf.initializers.variables(variables_to_init))
        print("Restoring parameters")
    except ValueError:
        sess.run(tf.initializers.global_variables())
        print("Initializing parameters")

    stats = actor_critic(env, policy_estimator, value_estimator, trainer, 10000, discount_factor=0.99, player2=True, positiveRewardFactor=1, negativeRewardFactor=1.2, batch_size=batch_size)

    filters = sess.run(conv_net.filter1)

    save_path = saver.save(sess, "tmp/model5.ckpt")
    print("Saving parameters")

end = time()

print("It took:", end-start, "seconds to do 5.000 games")

plotting.plot_episode_stats(stats, filters, smoothing_window=10)

