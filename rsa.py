import theano.tensor as T
import operator
from lasagne.layers import get_output

from bt import config
from neural import SimpleLasagneModel, NeuralLearner
from speaker import SpeakerLearner
from listener import ListenerLearner

parser = config.get_options_parser()
parser.add_argument('--rsa_listeners', type=int, default=1,
                    help='Number of listeners to use in RSA cooperative nets graph')
parser.add_argument('--rsa_speakers', type=int, default=1,
                    help='Number of speakers to use in RSA cooperative nets graph')
parser.add_argument('--eval_agent', type=int, default=0,
                    help='Index of the agent (listener/speaker) to use as the primary object'
                         'of evaluation. Whether this agent is a listener or speaker will be '
                         'inferred from the --listener flag.')

parser.add_argument('--rsa_alpha', type=float, nargs='*', default=[1.0],
                    help='Weights for the log-likelihood of the dataset according to the '
                         'listeners. Provide as many values as there are listeners.')
parser.add_argument('--rsa_beta', type=float, nargs='*', default=[1.0],
                    help='Weights for the log-likelihood of the dataset according to the '
                         'speakers. Provide as many values as there are speakers.')
parser.add_argument('--rsa_mu', type=float, nargs='*', default=[1.0],
                    help='Weights for KL(L_j||S_k). Provide values to fill a '
                         'num_listeners x num_speakers matrix, in row-major order '
                         '(i.e. all speakers for first listener, then all speakers for second '
                         'listener, etc.).')
parser.add_argument('--rsa_nu', type=float, nargs='*', default=[1.0],
                    help='Weights for KL(S_k||L_j). Provide values to fill a '
                         'num_listeners x num_speakers matrix, in row-major order '
                         '(i.e. all speakers for first listener, then all speakers for second '
                         'listener, etc.).')

parser.add_argument('--listener_samples', type=int, default=128,
                    help='Number of samples to draw from the listener per minibatch.')
parser.add_argument('--speaker_samples', type=int, default=128,
                    help='Number of samples to draw from the speaker per minibatch.')


class RSASubModel(object):
    '''
    A stand-in for the SimpleLasagneModel for the subcomponents of an RSA graph.
    '''
    def __init__(self, input_vars, target_var, l_out, loss, optimizer):
        self.input_vars = input_vars
        self.target_var = target_var
        self.l_out = l_out
        self.base_loss = loss
        self.optimizer = optimizer

    def build_sample_vars(self, num_other_agents):
        self.sample_inputs_self = [v.type('%s_sample_self' % v.name) for v in self.input_vars]
        self.sample_inputs_others = [[v.type('%s_sample_other%d' % (v.name, i))
                                      for v in self.input_vars]
                                     for i in range(num_other_agents)]
        t = self.target_var
        self.sample_target_self = t.type('%s_sample_self' % t.name)
        self.sample_target_others = [t.type('%s_sample_other%d' % (t.name, i))
                                     for i in range(num_other_agents)]

        self.all_synth_vars = (self.sample_inputs_self +
                               [self.sample_target_self] +
                               [v
                                for o_inputs, o_target in zip(self.sample_inputs_others,
                                                              self.sample_target_others)
                                for v in o_inputs + [o_target]])

    def data_to_synth_arrays(self, agent, samples_self, samples_others):
        def flatten(arrays):
            inputs, target = arrays
            return list(inputs) + [target]

        return [arr
                for samples in [samples_self] + samples_others
                for arr in flatten(agent._data_to_arrays(samples))]


class RSAGraphModel(SimpleLasagneModel):
    def __init__(self, listeners, speakers):
        self.listeners = listeners
        self.speakers = speakers
        input_vars = ([v for listener in listeners for v in listener.input_vars] +
                      [v for speaker in speakers for v in speaker.input_vars])
        target_vars = ([listener.target_var for listener in listeners] +
                       [speaker.target_var for speaker in speakers])
        super(RSAGraphModel, self).__init__(input_vars, target_vars,
                                            l_out=None, loss=None, optimizer=None)

    def get_train_loss(self, loss, l_out, target_var, params):
        for agent in self.speakers:
            agent.model.build_sample_vars(len(self.listeners))
        for agent in self.listeners:
            agent.model.build_sample_vars(len(self.speakers))

        est_loss = self.get_est_loss()
        est_grad = self.get_est_grad(params)
        synth_vars = [v
                      for agent in self.listeners + self.speakers
                      for v in agent.all_synth_vars]

        return est_loss, est_grad, synth_vars

    def get_est_loss(self):
        options = config.options()

        def loss_out(agent, out, target_var):
            pred = get_output(out)
            return agent.loss(pred, target_var)

        def kl(agent_p, agent_q, other_idx):
            return est_mean(agent_p.log_joint_emp(agent_p.sample_inputs_self,
                                                  agent_p.sample_target_self) -
                            agent_q.log_joint_smooth(agent_q.sample_inputs_others[other_idx],
                                                     agent_q.sample_target_others[other_idx]))

        # \alpha * KL(dataset || L) = \alpha * log L(dataset) + C
        est_loss = t_sum(alpha *
                         est_mean(loss_out(listener, listener.l_out, listener.model.target_var))
                         for alpha, listener in zip(options.rsa_alpha, self.listeners))
        # \beta * KL(dataset || S) = \beta * log S(dataset) + C
        est_loss += t_sum(beta *
                          est_mean(loss_out(speaker, speaker.l_out, speaker.model.target_var))
                          for beta, speaker in zip(options.rsa_beta, self.speakers))

        # \mu * KL(L || S)
        est_loss += t_sum(mu *
                          kl(listener, speaker, k)
                          for mu, (listener, j, speaker, k) in zip(options.rsa_mu, self.dyads()))
        # \nu * KL(S || L)
        est_loss += t_sum(nu *
                          kl(speaker, listener, j)
                          for nu, (listener, j, speaker, k) in zip(options.rsa_nu, self.dyads()))

        return est_loss

    def get_est_grad(self, params):
        options = config.options()

        def loss_grad(agent, out, target_var):
            pred = get_output(out)
            loss = agent.loss(pred, target_var)
            return T.grad(loss, params)

        # alpha and beta: train the agents directly against the dataset.
        #   \alpha_j E_D [-d/d\theta_j log L(c | m; \theta_j)]
        est_grad = t_sum(alpha *
                         est_mean(loss_grad(listener, listener.l_out, listener.model.target_var))
                         for alpha, listener in zip(options.rsa_alpha, self.listeners))
        #   \beta_k E_D [-d/d\phi_k log S(m | c; \phi_k)]
        est_grad += t_sum(beta *
                          est_mean(loss_grad(speaker, speaker.l_out, speaker.model.target_var))
                          for beta, speaker in zip(options.rsa_beta, self.speakers))

        # The "simple" mu and nu terms: train the agents directly against each other.
        # These are still ordinary log-likelihood terms; the complexity comes from
        # identifying the right input variables and iterating over the m x n dyads.
        #   sum_k \nu_jk E_{G_S(\phi_k)} [-d/d\theta_j log L(c | m; \theta_j)]
        est_grad += t_sum(nu *
                          est_mean(
                              loss_grad(listener,
                                        listener._get_l_out(listener.model.sample_inputs_others[k]),
                                        listener.model.sample_target_others[k])
                          )
                          for nu, (listener, j, speaker, k) in zip(options.rsa_nu, self.dyads()))
        #   sum_j \nu_jk E_{G_L(\theta_j)} [-d/d\phi_k log S(m | c; \phi_k)]
        est_grad += t_sum(mu *
                          est_mean(
                              loss_grad(speaker,
                                        speaker._get_l_out(speaker.model.sample_inputs_others[j]),
                                        speaker.model.sample_target_others[j])
                          )
                          for mu, (listener, j, speaker, k) in zip(options.rsa_mu, self.dyads()))

        # The "hard" mu and nu terms: regularize the agents with maximum entropy and
        # accommodating other agents' priors.
        #   sum_k \mu_jk E_{G_L(\theta_j)}
        #     [(1 + log G_L(c, m; \theta_j) - log H_S(c, m; \phi_k)) *
        #      d/d\theta_j log L(c | m; \theta_j)]
        est_grad += t_sum(mu *
                          est_mean(
                              (1 + listener.log_joint_emp(listener.model.sample_inputs_self) -
                               speaker.log_joint_smooth(speaker.model.sample_inputs_others[j])) *
                              loss_grad(listener,
                                        listener._get_l_out(listener.model.sample_inputs_self),
                                        speaker.model.sample_target)
                          )
                          for mu, (listener, j, speaker, k) in zip(options.rsa_mu, self.dyads()))
        #   sum_j \nu_jk E_{G_S(\phi_k)}
        #     [(1 + log G_S(c, m; \phi_k) - log H_L(c, m; \theta_j)) *
        #      d/d\phi_k log S(m | c; \phi_k)]
        est_grad += t_sum(nu *
                          est_mean(
                              (1 + speaker.log_joint_emp(speaker.model.sample_inputs_self) -
                               listener.log_joint_smooth(listener.model.sample_inputs_others[k])) *
                              loss_grad(speaker,
                                        speaker._get_l_out(speaker.model.sample_inputs_self),
                                        listener.model.sample_target)
                          )
                          for nu, (listener, j, speaker, k) in zip(options.rsa_nu, self.dyads()))

        return est_grad

    def dyads(self):
        for j, listener in enumerate(self.listeners):
            for k, speaker in enumerate(self.speakers):
                yield (listener, j, speaker, k)

    def minibatches(self, inputs, targets, batch_size, shuffle=False):
        raise NotImplementedError
        options = config.options()

        dataset = super(RSAGraphModel, self).minibatches(inputs, targets, batch_size,
                                                         shuffle=shuffle)
        for inputs_batch, targets_batch, _synth in dataset:
            synth_batch = []

            listener_samples = [listener.sample_joint_emp(options.listener_samples)
                                for listener in self.listeners]
            speaker_samples = [speaker.sample_joint_emp(options.speaker_samples)
                               for speaker in self.speakers]

            for listener, samples in zip(self.listeners, listener_samples):
                synth_batch.extend(listener.model.data_to_synth_arrays(listener, samples,
                                                                       speaker_samples))
            for speaker, samples in zip(self.speakers, speaker_samples):
                synth_batch.extend(speaker.model.data_to_synth_arrays(speaker, samples,
                                                                      listener_samples))
            yield inputs_batch, targets_batch, synth_batch


class RSALearner(NeuralLearner):
    def __init__(self):
        options = config.options()

        self.listeners = [ListenerLearner() for _ in range(options.num_listeners)]
        self.speakers = [SpeakerLearner() for _ in range(options.num_speakers)]

        agents = self.listeners if options.listener else self.speakers
        self.eval_agent = agents[options.eval_agent]

    def train(self, training_instances):
        options = config.options()

        if options.listener:
            listener_dataset = training_instances
            speaker_dataset = [inst.inverted() for inst in training_instances]
        else:
            listener_dataset = [inst.inverted() for inst in training_instances]
            speaker_dataset = training_instances

        for listener in self.listeners:
            listener._build_model(RSASubModel)
            listener.dataset = listener_dataset
        for speaker in self.speakers:
            speaker._build_model(RSASubModel)
            speaker.dataset = speaker_dataset

        return super(RSALearner, self).train(training_instances)

    def predict(self, eval_instances):
        return self.eval_agent.predict(eval_instances)

    def score(self, eval_instances):
        return self.eval_agent.score(eval_instances)

    def predict_and_score(self, eval_instances):
        return self.eval_agent.predict_and_score(eval_instances)

    def _data_to_arrays(self, training_instances):
        input_arrays = []
        target_arrays = []

        for agent in self.listeners + self.speakers:
            inputs, target = agent._data_to_arrays(training_instances)
            input_arrays.extend(inputs)
            target_arrays.append(target)

        return input_arrays, target_arrays

    def _build_model(self):
        self.model = RSAGraphModel(self.listeners, self.speakers)


def t_sum(seq):
    '''A version of sum that doesn't start with 0, for constructing
    Theano graphs without superfluous TensorConstants.'''
    return reduce(operator.add, seq)


def est_mean(vals):
    # TODO: control variates?
    return vals.mean()
