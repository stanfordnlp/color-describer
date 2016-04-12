import operator
import theano.tensor as T
from collections import OrderedDict

from stanza.unstable import config
from neural import SimpleLasagneModel, NeuralLearner
from vectorizers import SequenceVectorizer, BucketsVectorizer
from neural import OPTIMIZERS
from listener import LISTENERS
from speaker import SPEAKERS

parser = config.get_options_parser()
parser.add_argument('--rsa_listeners', type=int, default=1,
                    help='Number of listeners to use in RSA cooperative nets graph')
parser.add_argument('--rsa_speakers', type=int, default=1,
                    help='Number of speakers to use in RSA cooperative nets graph')
parser.add_argument('--listener_class', default='Listener', choices=LISTENERS.keys(),
                    help='The name of the listener model to use in the RSA network.')
parser.add_argument('--speaker_class', default='Speaker', choices=SPEAKERS.keys(),
                    help='The name of the speaker model to use in the RSA network.')
parser.add_argument('--eval_agent', type=int, default=0,
                    help='Index of the agent (listener/speaker) to use as the primary object '
                         'of evaluation. Whether this agent is a listener or speaker will be '
                         'inferred from the --listener flag.')
parser.add_argument('--rsa_optimizer', choices=OPTIMIZERS.keys(), default='rmsprop',
                    help='The optimization (update) algorithm to use for RSA training.')
parser.add_argument('--rsa_learning_rate', type=float, default=0.1,
                    help='The learning rate to use for RSA training.')

parser.add_argument('--rsa_alpha', type=float, nargs='*', default=[1.0],
                    help='Weights for the log-likelihood of the dataset according to the '
                         'listeners. Provide as many values as there are listeners.')
parser.add_argument('--rsa_beta', type=float, nargs='*', default=[1.0],
                    help='Weights for the log-likelihood of the dataset according to the '
                         'speakers. Provide as many values as there are speakers.')
parser.add_argument('--rsa_mu', type=float, nargs='*', default=[1.0],
                    help='Weights for KL(L_j||S_k). Provide values to fill a '
                         'rsa_listeners x rsa_speakers matrix, in row-major order '
                         '(i.e. all speakers for first listener, then all speakers for second '
                         'listener, etc.).')
parser.add_argument('--rsa_nu', type=float, nargs='*', default=[1.0],
                    help='Weights for KL(S_k||L_j). Provide values to fill a '
                         'rsa_listeners x rsa_speakers matrix, in row-major order '
                         '(i.e. all speakers for first listener, then all speakers for second '
                         'listener, etc.).')

parser.add_argument('--listener_samples', type=int, default=128,
                    help='Number of samples to draw from the listener per minibatch.')
parser.add_argument('--speaker_samples', type=int, default=128,
                    help='Number of samples to draw from the speaker per minibatch.')

parser.add_argument('--monitor_sublosses', action='store_true',
                    help='If `True`, return sub-losses for monitoring and write them to the '
                         'TensorBoard events file. This will likely increase compilation time.')
parser.add_argument('--monitor_subgrads', action='store_true',
                    help='If `True`, return sub-gradients for monitoring and write them to the '
                         'TensorBoard events file. This will likely increase compilation time.')
parser.add_argument('--grad_of_est', action='store_true',
                    help='If `True`, optimize using the gradient of the estimated loss; '
                         'otherwise, use the manually-derived estimate of the gradient of '
                         'the true loss.')
parser.add_argument('--layer_by_layer', action='store_true',
                    help='If `True`, train RSA agents layer-by-layer (only use the log-likelihood '
                         'sub-gradients, equivalent to training each agent on data generated from '
                         'the other agents); otherwise, use the gradient of the full RSA '
                         'objective.')
parser.add_argument('--listener_sample_smoothed', action='store_true',
                    help='If `True`, take samples from the smoothed utterance prior; otherwise, '
                         'sample from the empirical utterance prior.')
parser.add_argument('--speaker_sample_smoothed', action='store_true',
                    help='If `True`, take samples from the smoothed world prior; otherwise, '
                         'sample from the empirical world prior.')


class AggregatePrior(object):
    def __init__(self, listeners, speakers, prior_name='prior_emp'):
        self.listeners = listeners
        self.speakers = speakers
        self.prior_name = prior_name

    def fit(self, xs, ys):
        input_idx = 0
        for agent, target in zip(self.listeners + self.speakers, ys):
            num_inputs = len(agent.model.input_vars)
            inputs = xs[input_idx:input_idx + num_inputs]
            getattr(agent, self.prior_name).fit(inputs, [target])
            input_idx += num_inputs

    def apply(self, input_vars):
        assert False, ("AggregatePrior.apply shouldn't be called; "
                       "only individual model priors are used in RSA coop nets model")


class RSASubModel(SimpleLasagneModel):
    '''
    A SimpleLasagneModel for a subcomponent of an RSA graph.
    '''
    def __init__(self, input_vars, target_vars, l_out, loss, optimizer,
                 learning_rate=0.001, id=None):
        super(RSASubModel, self).__init__(input_vars, target_vars, l_out, loss, optimizer,
                                          learning_rate=learning_rate, id=id)
        if len(target_vars) != 1:
            raise ValueError('target_vars should be a sequence of length 1, instead got %s' %
                             (target_vars,))
        self.target_var = target_vars[0]

    def build_sample_vars(self, num_other_agents):
        self.sample_inputs_self = [v.type('%s_sample_self' % (v.name,))
                                   for v in self.input_vars]
        self.sample_inputs_others = [[v.type('%s_sample_other%d' % (v.name, i))
                                      for v in self.input_vars]
                                     for i in range(num_other_agents)]
        t = self.target_var
        self.sample_target_self = t.type('%s_sample_self' % (t.name,))
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
            inputs, targets = arrays
            return inputs + targets

        return [arr
                for i, samples in enumerate([samples_self] + samples_others)
                for arr in flatten(agent._data_to_arrays(samples, inverted=(i != 0)))]


class RSAGraphModel(SimpleLasagneModel):
    def __init__(self, listeners, speakers, eval_agent, id=None):
        options = config.options()

        self.listeners = listeners
        self.speakers = speakers
        self.eval_agent = eval_agent
        input_vars = ([v for listener in listeners for v in listener.model.input_vars] +
                      [v for speaker in speakers for v in speaker.model.input_vars])
        target_vars = ([listener.model.target_var for listener in listeners] +
                       [speaker.model.target_var for speaker in speakers])
        super(RSAGraphModel, self).__init__(input_vars, target_vars,
                                            l_out=eval_agent.model.l_out,
                                            loss=None, optimizer=OPTIMIZERS[options.rsa_optimizer],
                                            learning_rate=options.rsa_learning_rate,
                                            id=id)

    def params(self):
        result = []
        for listener in self.listeners:
            result.extend(listener.params())
        for speaker in self.speakers:
            result.extend(speaker.params())
        return result

    def get_train_loss(self, target_vars, params):
        options = config.options()

        for agent in self.speakers:
            agent.model.build_sample_vars(len(self.listeners))
        for agent in self.listeners:
            agent.model.build_sample_vars(len(self.speakers))

        monitored = self.get_est_loss()
        if options.grad_of_est:
            est_grad, monitored_grads = self.get_grad_of_est(monitored, params)
        else:
            est_grad, monitored_grads = self.get_est_grad(params,
                                                          layer_by_layer=options.layer_by_layer)
        monitored.update(monitored_grads)
        synth_vars = [v
                      for agent in self.listeners + self.speakers
                      for v in agent.model.all_synth_vars]

        return monitored, est_grad, synth_vars

    def get_est_loss(self):
        options = config.options()

        def kl(agent_p, agent_q, other_idx):
            return (
                agent_p.log_joint_emp(agent_p.model.sample_inputs_self,
                                      agent_p.model.sample_target_self) -
                agent_q.log_joint_smooth(agent_q.model.sample_inputs_others[other_idx],
                                         agent_q.model.sample_target_others[other_idx])
            ).mean()

        id_tag_log = (self.id + ': ') if self.id else ''
        id_tag = (self.id + '/') if self.id else ''
        # \alpha * KL(dataset || L) = \alpha * log L(dataset) + C
        if options.verbosity >= 4:
            print(id_tag_log + 'loss: KL(dataset || L)')
        alpha_losses = [
            ('%salpha_%s' % (id_tag, listener.id), alpha * listener.loss_out().mean())
            for alpha, listener in zip(options.rsa_alpha, self.listeners)
        ]
        # \beta * KL(dataset || S) = \beta * log S(dataset) + C
        if options.verbosity >= 4:
            print(id_tag_log + 'loss: KL(dataset || S)')
        beta_losses = [
            ('%sbeta_%s' % (id_tag, speaker.id), beta * speaker.loss_out().mean())
            for beta, speaker in zip(options.rsa_beta, self.speakers)
        ]

        # \mu * KL(L || S)
        if options.verbosity >= 4:
            print(id_tag_log + 'loss: KL(L || S)')
        mu_losses = [
            ('%smu_%s_%s' % (id_tag, listener.id, speaker.id), mu * kl(listener, speaker, j))
            for mu, (listener, j, speaker, k) in zip(options.rsa_mu, self.dyads())
        ]
        # \nu * KL(S || L)
        if options.verbosity >= 4:
            print(id_tag_log + 'loss: KL(S || L)')
        nu_losses = [
            ('%snu_%s_%s' % (id_tag, speaker.id, listener.id), nu * kl(speaker, listener, k))
            for nu, (listener, j, speaker, k) in zip(options.rsa_nu, self.dyads())
        ]

        all_sublosses = alpha_losses + beta_losses + mu_losses + nu_losses
        est_loss = t_sum(loss for tag, loss in all_sublosses)

        monitored = OrderedDict([('loss', est_loss)])
        if options.monitor_sublosses:
            monitored.update(all_sublosses)
        return monitored

    def get_est_grad(self, params, layer_by_layer=False):
        options = config.options()

        def mean_weighted_grad(weights, loss):
            # Lop to the rescue! Here I was calling T.jacobian and trying to
            # broadcast things and elementwise-multiply through the resulting lists,
            # when a function already existed to do all of that for me...
            return T.Lop(loss, params, weights / T.cast(weights.shape[0], 'float32'),
                         disconnected_inputs='ignore')
            # TODO: control variates?

        def mean_grad(loss):
            return T.grad(loss.mean(), params, disconnected_inputs='ignore')

        id_tag = (self.id + ': ') if self.id else ''
        # alpha and beta: train the agents directly against the dataset.
        #   \alpha_j E_D [-d/d\theta_j log L(c | m; \theta_j)]
        if options.verbosity >= 4:
            print(id_tag + 'grad: alpha')
        all_subgrads = [
            ('grad_alpha/%s' % (listener.id,),
             mean_grad(alpha * listener.loss_out()))
            for alpha, listener in zip(options.rsa_alpha, self.listeners)
        ]
        #   \beta_k E_D [-d/d\phi_k log S(m | c; \phi_k)]
        if options.verbosity >= 4:
            print(id_tag + 'grad: beta')
        all_subgrads.extend([
            ('grad_beta/%s' % (speaker.id,),
             mean_grad(beta * speaker.loss_out()))
            for beta, speaker in zip(options.rsa_beta, self.speakers)
        ])

        # The "simple" mu and nu terms: train the agents directly against each other.
        # These are still ordinary log-likelihood terms; the complexity comes from
        # identifying the right input variables and iterating over the m x n dyads.
        #   sum_k \nu_jk E_{G_S(\phi_k)} [-d/d\theta_j log L(c | m; \theta_j)]
        if options.verbosity >= 4:
            print(id_tag + 'grad: nu co-training')
        all_subgrads.extend([
            ('grad_nu_co/%s_%s' % (listener.id, speaker.id),
             mean_grad(nu * listener.loss_out(listener.model.sample_inputs_others[k],
                                              listener.model.sample_target_others[k])))
            for nu, (listener, j, speaker, k) in zip(options.rsa_nu, self.dyads())
        ])
        #   sum_j \nu_jk E_{G_L(\theta_j)} [-d/d\phi_k log S(m | c; \phi_k)]
        if options.verbosity >= 4:
            print(id_tag + 'grad: mu co-training')
        all_subgrads.extend([
            ('grad_mu_co/%s_%s' % (listener.id, speaker.id),
             mean_grad(mu * speaker.loss_out(speaker.model.sample_inputs_others[j],
                                             speaker.model.sample_target_others[j])))
            for mu, (listener, j, speaker, k) in zip(options.rsa_mu, self.dyads())
        ])

        # The "hard" mu and nu terms: regularize the agents with maximum entropy and
        # accommodating other agents' priors.
        #
        # Zero out these subgradients if we're doing layer-by-layer training.
        if not layer_by_layer:
            #   sum_k \mu_jk E_{G_L(\theta_j)}
            #     [(1 + log G_L(c, m; \theta_j) - log H_S(c, m; \phi_k)) *
            #      d/d\theta_j log L(c | m; \theta_j)]
            if options.verbosity >= 4:
                print(id_tag + 'grad: mu regularizer')
            all_subgrads.extend([
                ('grad_mu_reg/%s_%s' % (listener.id, speaker.id),
                 mean_weighted_grad(
                     mu *
                     (1 + listener.log_joint_emp(listener.model.sample_inputs_self,
                                                 listener.model.sample_target_self) -
                      speaker.log_joint_smooth(speaker.model.sample_inputs_others[j],
                                               speaker.model.sample_target_others[j])),
                     listener.loss_out(listener.model.sample_inputs_self,
                                       listener.model.sample_target_self)))
                for mu, (listener, j, speaker, k) in zip(options.rsa_mu, self.dyads())
            ])
            #   sum_j \nu_jk E_{G_S(\phi_k)}
            #     [(1 + log G_S(c, m; \phi_k) - log H_L(c, m; \theta_j)) *
            #      d/d\phi_k log S(m | c; \phi_k)]
            if options.verbosity >= 4:
                print(id_tag + 'grad: nu regularizer')
            all_subgrads.extend([
                ('grad_nu_reg/%s_%s' % (listener.id, speaker.id),
                 mean_weighted_grad(
                     nu *
                     (1 + speaker.log_joint_emp(speaker.model.sample_inputs_self,
                                                speaker.model.sample_target_self) -
                      listener.log_joint_smooth(listener.model.sample_inputs_others[k],
                                                listener.model.sample_target_others[k])),
                     speaker.loss_out(speaker.model.sample_inputs_self,
                                      speaker.model.sample_target_self)))
                for nu, (listener, j, speaker, k) in zip(options.rsa_nu, self.dyads())
            ])

        est_grad = t_sum([grads for tag, grads in all_subgrads], nested=True)

        monitored = OrderedDict()
        if options.monitor_grads:
            monitored.update([
                ('grad/' + param.name, grad)
                for param, grad in zip(params, est_grad)
            ])
        if options.monitor_subgrads:
            monitored.update([
                (tag + '/' + param.name, grad)
                for tag, grads in all_subgrads
                for param, grad in zip(params, grads)
            ])
        return est_grad, monitored

    def get_grad_of_est(self, monitored, params):
        options = config.options()

        grad_of_est = T.grad(monitored['loss'], params)

        monitored_grads = OrderedDict()
        if options.monitor_grads:
            monitored_grads.update([
                ('grad/' + param.name, grad)
                for param, grad in zip(params, grad_of_est)
            ])
        if options.monitor_subgrads:
            monitored_grads.update([
                (tag + '/' + param.name, grad)
                for tag, subloss in monitored.iteritems() if tag != 'loss'
                for param, grad in zip(params, T.grad(subloss, params,
                                                      disconnected_inputs='ignore'))
            ])

        return grad_of_est, monitored_grads

    def dyads(self):
        for j, listener in enumerate(self.listeners):
            for k, speaker in enumerate(self.speakers):
                yield (listener, j, speaker, k)

    def minibatches(self, inputs, targets, batch_size, shuffle=False):
        options = config.options()

        agents = self.listeners + self.speakers
        batches = super(RSAGraphModel, self).minibatches(inputs, targets, batch_size,
                                                         shuffle=shuffle)
        for dataset_inputs, dataset_targets, _synth in batches:
            inputs_batch = []
            targets_batch = []
            synth_batch = []

            filtered = self.filter_arrays(dataset_inputs, dataset_targets)
            for agent, (agent_inputs, agent_targets) in zip(agents, filtered):
                inputs_batch.extend(agent_inputs)
                targets_batch.extend(agent_targets)
                input_types = [a.shape for a in agent_inputs]
                target_types = [a.shape for a in agent_targets]
                if options.verbosity >= 8:
                    print('%s: %s -> %s' % (agent.id, input_types, target_types))

            listener_samples = [listener.sample_joint_smooth(options.listener_samples)
                                if options.listener_sample_smoothed else
                                listener.sample_joint_emp(options.listener_samples)
                                for listener in self.listeners]
            speaker_samples = [speaker.sample_joint_smooth(options.speaker_samples)
                               if options.speaker_sample_smoothed else
                               speaker.sample_joint_emp(options.listener_samples)
                               for speaker in self.speakers]

            for listener, samples in zip(self.listeners, listener_samples):
                arrays = listener.model.data_to_synth_arrays(listener, samples,
                                                             speaker_samples)
                synth_batch.extend(arrays)
                synth_types = [a.shape for a in arrays]
                if options.verbosity >= 8:
                    print('%s synth: %s' % (listener.id, synth_types))
            for speaker, samples in zip(self.speakers, speaker_samples):
                arrays = speaker.model.data_to_synth_arrays(speaker, samples,
                                                            listener_samples)
                synth_batch.extend(arrays)
                synth_types = [a.shape for a in arrays]
                if options.verbosity >= 8:
                    print('%s synth: %s' % (speaker.id, synth_types))
            yield inputs_batch, targets_batch, synth_batch

    def filter_arrays(self, inputs, targets):
        result = []
        input_idx = 0
        for agent, target in zip(self.listeners + self.speakers, targets):
            assert input_idx + len(agent.model.input_vars) <= len(inputs), \
                (input_idx, len(agent.model.input_vars), len(inputs))
            agent_inputs = inputs[input_idx:input_idx + len(agent.model.input_vars)]
            agent_targets = [target]
            result.append((agent_inputs, agent_targets))
            input_idx += len(agent.model.input_vars)
        return result


class RSALearner(NeuralLearner):
    def __init__(self, id=None):
        self.init_submodels(id)
        super(RSALearner, self).__init__(id=id)

        options = config.options()
        color_resolution = (options.listener_color_resolution
                            if options.listener else
                            options.speaker_color_resolution)
        self.seq_vec = SequenceVectorizer()
        self.color_vec = BucketsVectorizer(color_resolution, hsv=options.speaker_hsv)

    def init_submodels(self, id=None):
        options = config.options()

        id_tag = (id + '/') if id else ''
        self.listeners = [LISTENERS[options.listener_class](id='%sL%d' % (id_tag, j))
                          for j in range(options.rsa_listeners)]
        self.speakers = [SPEAKERS[options.speaker_class](id='%sS%d' % (id_tag, k))
                         for k in range(options.rsa_speakers)]

        agents = self.listeners if options.listener else self.speakers
        self.eval_agent = agents[options.eval_agent]

    def predict(self, eval_instances):
        return self.eval_agent.predict(eval_instances)

    def score(self, eval_instances):
        return self.eval_agent.score(eval_instances)

    def predict_and_score(self, eval_instances):
        return self.eval_agent.predict_and_score(eval_instances)

    def on_iter_end(self, step, writer):
        for agent in self.speakers + self.listeners:
            agent.on_iter_end(step, writer)

    def _data_to_arrays(self, training_instances,
                        init_vectorizer=False, test=False, inverted=False):
        options = config.options()

        input_arrays = []
        target_arrays = []

        if options.listener != inverted:
            listener_dataset = training_instances
            speaker_dataset = [inst.inverted() for inst in training_instances]
        else:
            listener_dataset = [inst.inverted() for inst in training_instances]
            speaker_dataset = training_instances

        for listener in self.listeners:
            if not test:
                listener.dataset = listener_dataset
            inputs, targets = listener._data_to_arrays(listener_dataset, test=test,
                                                       init_vectorizer=init_vectorizer)
            input_arrays.extend(inputs)
            target_arrays.extend(targets)
        for speaker in self.speakers:
            if not test:
                speaker.dataset = speaker_dataset
            inputs, targets = speaker._data_to_arrays(speaker_dataset, test=test,
                                                      init_vectorizer=init_vectorizer)
            input_arrays.extend(inputs)
            target_arrays.extend(targets)

        return input_arrays, target_arrays

    def _build_model(self):
        for agent in self.listeners + self.speakers:
            agent._build_model(RSASubModel)
        self.build_aggregate_model()

    def build_aggregate_model(self):
        self.model = RSAGraphModel(self.listeners, self.speakers, self.eval_agent)
        self.prior_emp = AggregatePrior(self.listeners, self.speakers, 'prior_emp')
        self.prior_smooth = AggregatePrior(self.listeners, self.speakers, 'prior_smooth')

    def __getstate__(self):
        return (self.seq_vec, self.color_vec,
                [agent.__getstate__() for agent in self.listeners + self.speakers])

    def __setstate__(self, state):
        self.seq_vec, self.color_vec, submodels = state
        self.init_submodels()
        for agent, substate in zip(self.listeners + self.speakers, submodels):
            agent.unpickle(substate, RSASubModel)
        self.build_aggregate_model()


def t_sum(seq, start=None, nested=False):
    '''A version of sum that doesn't start with 0, for constructing
    Theano graphs without superfluous TensorConstants.

    If `nested` is True, sum expressions embedded within lists,
    elementwise (for use with the output for T.jacobian).

    >>> t_sum([1, 2, 3])
    6
    >>> t_sum(xrange(1, 4), start=4)
    10
    >>> t_sum([[1, 2], [3, 4], [5, 6]], nested=True)
    [9, 12]
    >>> t_sum([[1, 2], [3, 4], [5, 6]], start=[-1, -2], nested=True)
    [8, 10]
    '''
    if nested:
        if not isinstance(seq, list):
            seq = list(seq)
        if start:
            return [t_sum(subseq, start_elem) for subseq, start_elem in zip(zip(*seq), start)]
        else:
            return [t_sum(subseq) for subseq in zip(*seq)]

    seq_list = list(seq)
    if seq_list:
        reduced = reduce(operator.add, seq_list)
        if start:
            reduced = start + reduced
        return reduced
    elif start:
        return start
    else:
        return 0
