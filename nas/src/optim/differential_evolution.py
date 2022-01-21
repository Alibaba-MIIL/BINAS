import sys

from scipy.optimize import differential_evolution
from scipy.optimize import NonlinearConstraint, LinearConstraint, Bounds



from nas.src.optim.block_frank_wolfe import BlockFrankWolfe
from nas.src.optim.utils import *

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=11)


class DiffEvo(BlockFrankWolfe):
    def __init__(self, params, list_alphas, inference_time_limit, max_gamma, **kwargs):
        super(DiffEvo, self).__init__(params, list_alphas, inference_time_limit, max_gamma)
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

    def set_linear_fitness(self, accuracy_vec, accuracy_vec_beta, linear):
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        alphas = np.sum(alpha_blocks)
        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks) - len(beta_blocks) * self.min_depth

        # Analytical Accuracy Predictor Objective
        _, pa, pb = self._alpha_beta_accuracy_matrix(alpha_blocks, beta_blocks, accuracy_vec, accuracy_vec_beta, linear)

        self.obj = -np.concatenate((pa, pb))


    def set_predictor(self, predictor):
        self.predictor = predictor
        self.predictor.model.cpu()
        self.predictor.eval()


    def evo(self):
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        alphas = np.sum(alpha_blocks)
        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks) - len(beta_blocks) * self.min_depth

        # Simplex Constraints
        A_eq, b_eq = self._simplex_eq_constraint(alpha_blocks + beta_blocks, alphas + betas)

        lc = LinearConstraint(A=A_eq, lb=b_eq, ub=b_eq, keep_feasible=True)

        lb = [0] * (alphas + betas)
        ub = [1] * (alphas + betas)
        bounds = Bounds(lb=lb, ub=ub, keep_feasible=True)

        # Latency Quadratic Constraint
        self._alpha_beta_latency_matrix(alpha_blocks, beta_blocks, latency_vec)

        nlc = NonlinearConstraint(fun=self.latency_con, lb=-np.inf, ub=self.T)

        # Init
        x0 = self.smallest_sol(self.generate_cnames(alpha_blocks, beta_blocks))

        if self.predictor is not None:
            fitness = predictor_fitness
            args = tuple([self.predictor])
        elif self.obj is not None:
            fitness = linear_fitness
            args = tuple([self.obj])
        else:
            raise Exception('Either linear fitness or predictor fitness is to be specified.')

        print(args)
        result = differential_evolution(func=fitness, bounds=bounds, args=args, constraints=(lc, nlc),
                                        x0=x0,
                                        # seed=1,
                                        disp=True,
                                        workers=4,
                                        # mutation=self.mutate_prob,
                                        popsize=self.population_size,
                                        # maxiter=self.max_time_budget,
                                        maxiter=10,
                                        # recombination=1-self.parent_ratio
                                        )

        x = result.x

        x = np.array(x)
        x = x.squeeze().clip(0, 1)

        update_attentions_inplace(self.list_alphas,
                                  alpha_attention_vec=x[:self.alphas],
                                  beta_attention_vec=x[-self.betas:])

        # print('EA alphas')
        # print(np.reshape(x[:alphas], (len(alpha_blocks), -1)))
        # print('EA betas')
        # print(np.reshape(x[-betas:], (len(beta_blocks), -1)))

def linear_fitness(x, *args):
    return np.dot(np.array(x), args)

def predictor_fitness(x, *args):
    with torch.no_grad():
        retval = args[0](x)

    retval = retval.cpu().numpy()
    return retval

