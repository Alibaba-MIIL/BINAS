import sys

import cplex

from nas.src.optim.block_frank_wolfe import BlockFrankWolfe
from nas.src.optim.utils import *

np.set_printoptions(threshold=sys.maxsize, suppress=True, precision=11)


class IQCP(BlockFrankWolfe):
    def __init__(self, params, list_alphas, inference_time_limit, max_gamma):
        super(IQCP, self).__init__(params, list_alphas, inference_time_limit, max_gamma)


    def smallest_sol(self, cnames):
        vals = [1.0 if (name[0] == 'a' and name.split('_')[-1] == '0') or
                       (name[0] == 'b' and name.split('_')[-1] == '1') else 0.0
                for name in cnames]

        # print(vals)
        return cplex.SparsePair(ind=range(len(cnames)), val=vals)


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


    def qcp(self, accuracy_vec, accuracy_vec_beta, linear):
        alpha_attention_vec, latency_vec, _, alpha_blocks, beta_attention_vec, _, beta_blocks = \
            flatten_attention_latency_grad_alpha_beta_blocks(self.list_alphas)

        alphas = np.sum(alpha_blocks)
        assert alphas == len(accuracy_vec)

        betas = np.sum(beta_blocks)
        assert betas == len(alpha_blocks) - len(beta_blocks) * self.min_depth

        # Analytical Accuracy Predictor Objective
        Q_acc, pa, pb = self._alpha_beta_accuracy_matrix(alpha_blocks, beta_blocks, accuracy_vec, accuracy_vec_beta,
                                                         linear)

        obj = np.concatenate((pa, pb))
        obj = -obj

        # Support full Q_obj of size (alphas + betas, alphas + betas) and also (alphas, betas)
        if Q_acc.shape[0] == alphas and Q_acc.shape[1] == betas:
            Q_acc_padded = np.zeros((alphas + betas, alphas + betas))
            Q_acc_padded[:alphas, -betas:] = Q_acc
            Q_acc = Q_acc_padded

        # Make sure the matrix is symmetric
        Q_obj = -(Q_acc + Q_acc.T) / 2 if Q_acc is not None else None

        # Double the matrix since CPLEX does so automatically:     1/2 x.T * Q_obj * x + p * x
        Q_obj = 2 * Q_obj if Q_obj is not None else None

        # cname = [f'a{i}' for i in range(alphas)] + [f'b{i}' for i in range(betas)]
        cname = self.generate_cnames(alpha_blocks, beta_blocks)
        # print(cname)

        # Simplex Constraints
        A_eq, b_eq = self._simplex_eq_constraint(alpha_blocks + beta_blocks, alphas + betas)

        lb = [0] * len(obj)
        ub = [1] * len(obj)

        rows = []
        for r in range(A_eq.shape[0]):
            ind = np.nonzero(A_eq[r, :])[0].tolist()
            cnames = [cname[i] for i in ind]
            coeffs = [1] * len(cnames)
            rows.append([cnames, coeffs])

        rhs = [1] * len(rows)
        sense = 'E' * len(rows)
        rname = [f'simplex_{i + 1}' for i in range(len(rows))]

        # Latency Quadratic Constraint
        Q, p = self._alpha_beta_latency_matrix(alpha_blocks, beta_blocks, latency_vec)

        ind = np.nonzero(p)[0]
        qlin = [cplex.SparsePair([cname[i] for i in ind.tolist()], p[ind].tolist())]

        vars_1, vars_2, coeffs = [], [], []
        for b in range(Q.shape[1]):
            for a in range(Q.shape[0]):
                if Q[a, b] == 0:
                    continue

                vars_1.append(cname[a])
                vars_2.append(cname[alphas + b])
                coeffs.append(Q[a, b])

        quad = [cplex.SparseTriple(vars_1, vars_2, coeffs)]
        qsense = ['L']
        qrhs = [self.T - self._fixed_latency]
        qname = ['latency']

        c = cplex.Cplex()
        c.variables.add(obj=obj, types='I' * len(cname), lb=lb, ub=ub, names=cname)

        # Support full Q_obj of size (alphas + betas, alphas + betas) and also (alphas, betas)
        offset = alphas if Q_obj.shape[0] == alphas else 0

        if Q_obj is not None:
            for a in range(Q_obj.shape[0]):
                for b in range(Q_obj.shape[1]):
                    if Q_obj[a, b] == 0:
                        continue

                    # c.objective.set_quadratic_coefficients(v1, v2, val)
                    # Note: Since the quadratic objective function must be symmetric, each triple in which v1 is different
                    # from v2 is used to set both the (v1, v2) coefficient and the(v2, v1) coefficient.
                    # If (v1, v2) and (v2, v1) are set with a single call, the second value is stored.
                    c.objective.set_quadratic_coefficients(cname[a], cname[offset + b], Q_obj[a, b])

        c.linear_constraints.add(lin_expr=rows, senses=sense, rhs=rhs, names=rname)
        for q in range(0, len(qname)):
            c.quadratic_constraints.add(lin_expr=qlin[q],
                                        quad_expr=quad[q],
                                        sense=qsense[q],
                                        rhs=qrhs[q],
                                        name=qname[q])

        # Set initial feasible solution
        c.MIP_starts.add(self.smallest_sol(cname), c.MIP_starts.effort_level.auto, "smallest")
        # print(c.MIP_starts.get_names())

        tol_conv = 1e-9
        tol_int = 1e-1
        c.parameters.barrier.qcpconvergetol.set(tol_conv)
        # c.parameters.mip.tolerances.integrality.set(tol_int)
        c.parameters.timelimit.set(1e2)

        c.solve()
        # print(f'T: {self.T}')
        if not c.solution.get_status() == c.solution.status.optimal and \
            not c.solution.get_status() == c.solution.status.MIP_optimal:
            print(f'WARNING: No optimal solution found. Status: {c.solution.get_status_string()}')

        x = c.solution.get_values()

        x = np.array(x)
        x = x.squeeze().clip(0, 1)

        update_attentions_inplace(self.list_alphas, alpha_attention_vec=x[:alphas], beta_attention_vec=x[-betas:])

