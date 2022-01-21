import sys

from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint, LinearConstraint, Bounds



from nas.src.optim.block_frank_wolfe import BlockFrankWolfe
from nas.src.optim.evo_algo import EvolutionFinder
from nas.src.optim.utils import *

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=11)


class Evo(BlockFrankWolfe):
    def __init__(self, params, list_alphas, inference_time_limit, max_gamma, **kwargs):
        super(Evo, self).__init__(params, list_alphas, inference_time_limit, max_gamma)

        self.mutate_prob = kwargs.get('mutate_prob', 0.1)
        self.population_size = kwargs.get('population_size', 100)
        self.max_time_budget = kwargs.get('max_time_budget', 500)
        self.parent_ratio = kwargs.get('parent_ratio', 0.25)
        self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

        _, _, _, alpha_blocks, _, _, beta_blocks = flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        self.alphas = np.sum(alpha_blocks)
        self.betas = np.sum(beta_blocks)
        self.obj = None
        self.predictor = None
        self.accuracy_base = None


    def smallest_sol(self, cnames):
        vals = [1.0 if (name[0] == 'a' and name.split('_')[-1] == '0') or
                       (name[0] == 'b' and name.split('_')[-1] == '1') else 0.0
                for name in cnames]

        return vals


    def generate_cnames(self, alpha_blocks, beta_blocks):
        aname = []
        bname = []
        alpha_offset = 0
        for beta_block, beta_block_size in enumerate(beta_blocks):
            aname += [f'a_{beta_block}_0_{c}' for c in range(alpha_blocks[alpha_offset])]
            alpha_offset += 1
            for b in range(1, beta_block_size + 1):
                bname.append(f'b_{beta_block}_{b}')
                aname += [f'a_{beta_block}_{b}_{c}' for c in range(alpha_blocks[alpha_offset])]
                alpha_offset += 1

        assert alpha_offset == len(alpha_blocks)

        return aname + bname


    def latency_con(self, x):
        res = self.latency_formula(x[:self.alphas], x[-self.betas:], self.fixed_latency)
        # print(res)
        # print(self.T)
        return res

    def fitness(self, x):
        if self.obj is not None:
            return np.dot(np.array(x), self.obj)

        if self.accuracy_base is not None:
            x = np.array(x)
            accs = []
            for i in range(x.shape[0]):
                alpha = x[i, :self.alphas]
                beta = x[i, -self.betas:]
                accs.append(self.predictor(alpha, beta, self.accuracy_base))

            return accs

        return self.predictor(x)

    def set_linear_fitness(self, accuracy_vec, accuracy_vec_beta, linear):
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        alphas = np.sum(alpha_blocks)
        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks) - len(beta_blocks) * self.min_depth

        # Analytical Accuracy Predictor Objective
        _, pa, pb = self._alpha_beta_accuracy_matrix(alpha_blocks, beta_blocks, accuracy_vec, accuracy_vec_beta, linear)

        self.obj = np.concatenate((pa, pb))


    def set_predictor(self, predictor, accuracy_base=None):
        self.accuracy_base = accuracy_base
        self.predictor = predictor


    def evo(self):
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        alphas = np.sum(alpha_blocks)
        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks) - len(beta_blocks) * self.min_depth

        # Latency Quadratic Constraint
        self._alpha_beta_latency_matrix(alpha_blocks, beta_blocks, latency_vec)

        # Init
        self.evo_finder = EvolutionFinder(efficiency_constraint=self.T,
                                          efficiency_predictor=self.latency_con,
                                          accuracy_predictor=self.fitness,
                                          constraint_type='PyTorch_CPU',
                                          blocks=alpha_blocks + beta_blocks)

        x = self.evo_finder.run_evolution_search()
        # print(x)

        x = np.array(x)
        x = x.squeeze().clip(0, 1)

        update_attentions_inplace(self.list_alphas, alpha_attention_vec=x[:alphas], beta_attention_vec=x[-betas:])
        return torch.tensor(x[:alphas]).float(), torch.tensor(x[-betas:]).float()
        # print('EA alphas')
        # print(np.reshape(x[:alphas], (len(alpha_blocks), -1)))
        # print('EA betas')
        # print(np.reshape(x[-betas:], (len(beta_blocks), -1)))

