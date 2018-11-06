import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from time import time

import sklearn.pipeline
import sklearn.preprocessing

from fourInARowWrapper import FourInARowWrapper

if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')

env = FourInARowWrapper(1)
#env.observation_space.sample()

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
#observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
#scaler = sklearn.preprocessing.StandardScaler()
#scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
#featurizer = sklearn.pipeline.FeatureUnion([
#        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
#        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
#        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
#        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
#        ])
#featurizer.fit(scaler.transform(observation_examples))

# def featurize_state(state):
#     """
#     Returns the featurized representation for a state.
#     """
#     scaled = scaler.transform([state])
#     featurized = featurizer.transform(scaled)
#     return featurized[0]

class ConvolutionalNetwork():
    def __init__(self, scope="conv_net"):
        with tf.variable_scope(scope):
            self.board = tf.placeholder(tf.float32, (7, 6, 2), "board")

            self.filter1 = tf.Variable(tf.random_normal([3, 3, 2, 40]), name="filter1")
            self.filter2 = tf.Variable(tf.random_normal([2, 2, 12, 30]), name="filter2")

            self.conv1 = tf.nn.conv2d(
                input=tf.expand_dims(self.board, 0),
                filter=self.filter1,
                strides=(1, 1, 1, 1),
                padding="SAME"
            )

            self.l1 = tf.nn.relu(self.conv1)

            # self.l1 = tf.layers.max_pooling2d(
            #     inputs = self.l1,
            #     pool_size = (2, 2),
            #     strides = (2, 2),
            #     padding = "VALID",
            #     data_format='channels_last',
            #     name="maxPooling"
            # )

            # self.conv2 = tf.nn.conv2d(
            #     input=self.l1,
            #     filter=self.filter2,
            #     strides=(1, 1, 1, 1),
            #     padding="SAME"
            # )

            self.outLayer = tf.nn.relu(self.conv1)

            self.outLayer = tf.layers.max_pooling2d(
                inputs = self.outLayer,
                pool_size = (2, 2),
                strides = (2, 2),
                padding = "VALID",
                data_format='channels_last',
                name="maxPooling"
            )

class Trainer():
    def __init__(self, learning_rate=0.001, policy=None, policyLossFactor=0.1, value=None, valueLossFactor=0.1, scope="trainer"):
        with tf.variable_scope(scope):
            self.policy = policy
            self.value = value
            self.loss = policyLossFactor * policy.loss + valueLossFactor * value.loss
            #self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def update(self, state, td_target, td_error, action, avaliableColumns, sess=None):
        sess = sess or tf.get_default_session()
        # state = featurize_state(state)

        player = state[0]
        board = state[1]

        feed_dict = {self.policy.board: board, self.policy.player: player, self.policy.target: td_error, self.policy.action: action,
                    self.policy.validColumnsFilter: avaliableColumns, self.value.board: board, self.value.player: player, self.value.target: td_target}
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, entropyFactor=0.1, shared_layers=None, scope="policy_estimator"):
        with tf.variable_scope(scope):
            if shared_layers is not None:
                self.board = shared_layers.board
                self.input = shared_layers.outLayer
            else:
                print("Needs shared_layers parameter")
                return -1

            self.player = tf.placeholder(tf.float32, (2,), "player")
            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.validColumnsFilter = tf.placeholder(dtype=tf.float32, shape=(7,), name="validColumnsFilter")

            self.input = tf.reshape(self.input, [1, 360])
            self.player_exp = tf.expand_dims(self.player, axis=0, name="player_exp")
            self.input = tf.concat([self.player_exp, self.input], 1)

            self.l1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=180,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            # self.l2 = tf.contrib.layers.fully_connected(
            #     inputs=self.l1,
            #     num_outputs=100,
            #     activation_fn=tf.nn.sigmoid,
            #     weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            # )

            self.mu = tf.contrib.layers.fully_connected(
                inputs=self.l1,
                num_outputs=env.action_space.high-env.action_space.low,
                activation_fn=tf.nn.softmax,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.mu = tf.squeeze(self.mu)

            self.mu = tf.multiply(self.mu, self.validColumnsFilter) + 1e-4

            self.dist = tf.contrib.distributions.Categorical(probs=self.mu, dtype=tf.float32)

            # Draw sample
            #self.action = self.normal_dist._sample_n(1)
            self.action = self.dist.sample()

            # Clip sample into allowed action space
            #self.action = tf.clip_by_value(self.action, env.action_space.low, env.action_space.high-1)

            # Loss and train op
            #self.loss = -self.normal_dist.log_prob(self.action) * self.target
            self.loss = -self.dist.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration
            #self.loss -= 1e-1 * self.normal_dist.entropy()
            self.loss -= entropyFactor * self.dist.entropy()

            #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # self.train_op = self.optimizer.minimize(
            #     self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, env, sess=None):
        sess = sess or tf.get_default_session()
        #state = featurize_state(state)

        player = env.state[0]
        board = env.state[1]

        #print("Player:", player, "Board", board)

        action, mu = sess.run([self.action, self.mu], {self.player: player, self.board: board, self.validColumnsFilter: env.getAvaliableColumns()})
        return action, mu

    # def update(self, state, target, action, avaliableColumns, sess=None):
    #     sess = sess or tf.get_default_session()
    #     #state = featurize_state(state)
    #
    #     player = state[0]
    #     board = state[1]
    #
    #     feed_dict = {self.board: board, self.player: player, self.target: target, self.action: action, self.validColumnsFilter: avaliableColumns}
    #     _, loss = sess.run([self.train_op, self.loss], feed_dict)
    #     return loss


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, learning_rate=0.1, shared_layers=None, scope="value_estimator"):
        with tf.variable_scope(scope):
            if shared_layers is not None:
                self.board = shared_layers.board
                self.input = shared_layers.outLayer
            else:
                print("Needs shared_layers parameter")
                return -1

            #self.board = tf.placeholder(tf.float32, [7,6,2], "board")
            self.player = tf.placeholder(tf.float32, (2,), "player")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            self.input = tf.reshape(self.input, [1,360])
            self.player_exp = tf.expand_dims(self.player, 0)
            self.input = tf.concat([self.player_exp, self.input], 1)

            self.l1 = tf.contrib.layers.fully_connected(
                inputs=self.input,
                num_outputs=180,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            # self.l2 = tf.contrib.layers.fully_connected(
            #     inputs=self.l1,
            #     num_outputs=100,
            #     activation_fn=tf.nn.sigmoid,
            #     weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            # )

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.l1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1))

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            #self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # self.train_op = self.optimizer.minimize(
            #     self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        player = state[0]
        board = state[1]
        #state = featurize_state(state)
        return sess.run(self.value_estimate, {self.board: board, self.player: player})

    # def update(self, state, target, sess=None):
    #     sess = sess or tf.get_default_session()
    #
    #     player = state[0]
    #     board = state[1]
    #
    #     #state = featurize_state(state)
    #     feed_dict = {self.board: board, self.player: player, self.target: target}
    #     _, loss = sess.run([self.train_op, self.loss], feed_dict)
    #     return loss


def actor_critic(env, estimator_policy, estimator_value, trainer, num_episodes, discount_factor=1.0, player2=True, positiveRewardFactor=1.0, negativeRewardFactor=1.0):
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

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_td_error=np.zeros(num_episodes),
        episode_value_loss=np.zeros(num_episodes),
        episode_policy_loss=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    game = 1

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset(i_episode % 2 + 1)

        player = state[0]
        board = state[1]

        episode = []

        player1_state = None
        player1_action = None

        player2_state = None
        player2_action = None
        probas = None
        last_turn = False
        done = False
        state_tmp = None
        last_state = None
        action_tmp = None
        reward_tmp = None
        action = None
        reward = None

        if game == num_episodes-3:
            player2 = False

        # One step in the environment
        for t in itertools.count():
            #print("------------------------------------------New loop----------------------------")
            # Save avaliable columns
            if not done:
                avaliableColumns = env.getAvaliableColumns()

            currentPlayerBeforeStep = env.getCurrentPlayer()

            action_tmp = action
            reward_tmp = reward

            # Take a step
            if currentPlayerBeforeStep == 1 or currentPlayerBeforeStep == 2 and player2 and not done:
                action, probas = estimator_policy.predict(env)
                # if currentPlayerBeforeStep == 2:
                #     action = int(np.random.randint(0, 7))
            elif not done:
                try:
                    action = int(input("Give a column number: ")) - 1
                except ValueError:
                    print("Wrong input! Setting action to 1")
                    action = 0
                probas = None

            if not done:
                next_state, reward, step_done, _ = env.step(action)

                next_player = next_state[0]
                next_board = next_state[1]



                if step_done:
                    pass
                    #print("step_done")
                    #print("Player", currentPlayerBeforeStep, "won!!!!")

                if t > 0:
                    state_tmp = last_state
                    last_state = state
                    reward_tmp = -reward*negativeRewardFactor
                else:
                    state_tmp = state
                    last_state = state
                    reward_tmp = -reward*negativeRewardFactor


            elif done and not last_turn:
                #print("done and not last turn")
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
                # print("player", player, "is playing\nReward:", episode[-1].reward, "\nAction:", episode[-1].action + 1)
                # print("State")
                # env.renderHotEncodedState(episode[-1].state)
                # print("Next State")
                # env.renderHotEncodedState(episode[-1].next_state)

                # Update statistics
                stats.episode_rewards[i_episode] += episode[-1].reward
                stats.episode_lengths[i_episode] = t

                # Calculate TD Target
                value_next = estimator_value.predict(episode[-1].next_state)
                td_target = episode[-1].reward + discount_factor * value_next


                td_error = td_target - estimator_value.predict(episode[-1].state)


                stats.episode_td_error[i_episode] = td_error

                # Update the value estimator
                #value_loss = estimator_value.update(episode[-1].state, td_target)

                # Update the policy estimator
                # using the td error as our advantage estimate
                #policy_loss = estimator_policy.update(episode[-1].state, td_error, episode[-1].action, avaliableColumns)

                # Update both networks
                loss = trainer.update(episode[-1].state, td_target, td_error, episode[-1].action, avaliableColumns)



                stats.episode_value_loss[i_episode] = loss
                #stats.episode_policy_loss[i_episode] = policy_loss

                # Print out which step we're on, useful for debugging.
                print("Player {}: Action {}, Reward {}, TD Error {}, Loss {} at Step {} @ Game {} @ Episode {}/{} ({})\n".format(
                    player , episode[-1].action+1, episode[-1].reward, td_error, loss, t, game, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

            if game == num_episodes or env.getCurrentPlayer() == 2 and not player2:
                env.render()
                if probas is not None:
                    out = " "
                    for i in range(probas.size):
                        out += "%.1f " % probas[i]
                    print(out)

            #if last_turn:

            #    break



            if done:
                #print("--------------------------------Game ended!!")
                last_turn = True
                game += 1


            if step_done:
                done = True

            state = next_state

    return stats

tf.reset_default_graph()

start = time()

global_step = tf.Variable(0, name="global_step", trainable=False)
conv_net = ConvolutionalNetwork()
policy_estimator = PolicyEstimator(entropyFactor=1e-4, shared_layers=conv_net)
value_estimator = ValueEstimator(learning_rate=0.001, shared_layers=conv_net)
trainer = Trainer(learning_rate=1e-3, policy=policy_estimator, policyLossFactor=1, value=value_estimator, valueLossFactor=1e-2)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    # Note, due to randomness in the policy the number of episodes you need varies
    # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
    stats = actor_critic(env, policy_estimator, value_estimator, trainer, 20000, discount_factor=0.99, player2=True, positiveRewardFactor=1, negativeRewardFactor=1)

    filters = sess.run(conv_net.filter1)


plotting.plot_episode_stats(stats, filters, smoothing_window=10)

end = time()

print("It took:", end-start, "seconds to do 5.000 games")
