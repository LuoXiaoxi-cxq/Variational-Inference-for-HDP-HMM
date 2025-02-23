import numpy as np
from tqdm import tqdm
from scipy.special import digamma

num_seq = 500
num_state = 10
num_obs = 500
max_L = 20


def Elog_dir(alpha: np.array):
    """
    Calculate E_q[log(\theta)], where theta_k is a vector that follows dirichlet distribution.
    :param alpha: parameter array of dirichlet distribution
    :return: array of E_q[log(\theta_{kw})]
    """
    return digamma(alpha) - digamma(np.sum(alpha, axis=1, keepdims=True))


def Elog_eta_k(alpha: np.array):
    """
    return E_q[log(\eta)], where ln(\eta_{km}) = ln(\eps_{km}) + \sum_{j<m}ln(1-\eta_{kj}), where \eps_{km} ~ Beta(a, b)
    :param alpha: array, alpha[k,m] are the parameters of \eps_{km}
    """
    # with shape (T0,T1,2), Elog_beta[:,:,0] for E_q[ln(\epsilon_km)], and Elog_beta[:,:,1] for E_q[ln(1-\epsilon_km)]
    Elog_beta = digamma(alpha) - digamma(np.sum(alpha, axis=2, keepdims=True))
    Elog_eta = np.copy(Elog_beta[:, :, 0])
    for i in range(1, T1):
        Elog_eta[:, i:] += Elog_beta[:, i - 1, 1][..., np.newaxis]
    return Elog_eta


def Elog_eta_0(alpha: np.array):
    """
    return E_q[log(\eta_0)], where ln(\eta_{0k}) = ln(\eps_{k}) + \sum_{j<k}ln(1-\eta_{j}), where \eps_{km} ~ Beta(c, d)
    :param alpha: array, alpha[k,m] are the parameters of \eps_{km}
    """
    # with shape (T0,T1,2), Elog_beta[:,:,0] for E_q[ln(\epsilon_km)], and Elog_beta[:,:,1] for E_q[ln(1-\epsilon_km)]
    Elog_beta = digamma(alpha) - digamma(np.sum(alpha, axis=1, keepdims=True))
    Elog_eta = np.copy(Elog_beta[:, 0])
    for k in range(1, T0):
        Elog_eta[k] += np.sum(Elog_beta[:k, 1])
    return Elog_eta


def FB(A: np.array, B: np.array, seq: np.array):
    """
    Forward-Backward algorithm for computing alpha_{i}(k) and beta_i(k) of the d-th sequence
    :param A: transition matrix
    :param B: emission matrix
    :param seq: observation
    :return: array of alpha_{di}(k) and beta_di(k)
    """
    pi = np.ones(T0) / T0  # uniform distribution
    seq_L = len(seq)
    assert seq.all() >= 0
    g_ls = list()  # scaling term

    # --- update alpha ---
    alpha = np.zeros((seq_L, T0))
    # \alpha_1(k) = \pi_k * B_{k,x_1}, for 1 \leq k \leq T0
    alpha[0, :] = pi * B[:, seq[0]]
    alpha_hat = np.zeros((seq_L, T0))
    g_1 = np.sum(alpha[0])
    alpha_hat[0, :] = alpha[0, :] / g_1
    g_ls.append(g_1)

    # \alpha_i(k) = p(x_i|theta_k) * \sum_{j=1}^T0 [\alpha_{i-1}(j) * A_{j,k}], for 1 \leq k \leq T0
    for i in range(1, seq_L):
        alpha[i, :] = B[:, seq[i]] * (alpha[i - 1, :].reshape((1, -1)) @ A)
        alpha_hat_tmp = B[:, seq[i]] * (alpha_hat[i - 1, :].reshape((1, -1)) @ A)
        g_i = np.sum(alpha_hat_tmp)
        alpha_hat[i, :] = alpha_hat_tmp / g_i
        g_ls.append(g_i)

    # --- update beta ---
    beta_hat = np.zeros((seq_L, T0))
    beta = np.zeros((seq_L, T0))
    beta[seq_L - 1, :] = 1
    beta_hat[seq_L - 1, :] = 1
    # \beta_i(k) = \sum_{j=1}^T0 [A_{kj} * p(x_{i+1}|theta_j) * \beta_{i+1}(j)], for 1 \leq k \leq T0
    for i in range(seq_L - 2, -1, -1):
        beta[i] = np.squeeze(A @ ((B[:, seq[i + 1]] * beta[i + 1, :]).reshape(-1, 1)))
        beta_hat_tmp = np.squeeze(A @ ((B[:, seq[i + 1]] * beta_hat[i + 1, :]).reshape(-1, 1)))
        beta_hat[i] = beta_hat_tmp / g_ls[i]

    # # p(z_i=k) = alpha_i(k) * beta_i(k) / sum_j(alpha_i(j) * beta_i(j))
    tmp_ab = alpha_hat * beta_hat + 1e-100
    # # the (i, k) element in p_z is p(z_i=k)
    p_z = tmp_ab / np.sum(tmp_ab, axis=1, keepdims=True)

    # wanted to use log and exp but failed
    # log_p_z = np.log(alpha) + np.log(beta) - np.log(np.sum(alpha * beta, axis=1, keepdims=True))
    # p_z = np.exp(log_p_z)
    return alpha_hat, beta, p_z


def VI(obs_arr: np.array, T0: int, T1: int, seq_N: int, seq_L: int, obs_N: int, b0=1, alpha_0=1, tau_0=1):
    """
    Variational Inference algorithm for HDP-HMM. Parameters theta, zeta, epsilon, c, z are inferred.
    :param obs_arr: Observations.
    :param T0: Truncate G to T0. The number of global hidden states.
    :param T1: Truncate G_k to T1.
    :param seq_N: the number of observed sequences. 50000
    :param seq_L: The maximum length of observed sequences. 30
    :param obs_N: the numebr of distinct observations.
    :return: Hidden states (z) of each observation.
    """
    u_theta = np.random.dirichlet(np.ones(obs_N), size=T0)
    u_zeta = np.ones((T0, 2))
    u_eps = np.ones((T0, T1, 2))
    u_c = np.random.dirichlet(np.ones(T0), size=(T0, T1))
    u_z = np.random.dirichlet(np.ones(T0), size=(seq_N, seq_L))
    old_theta, old_zeta, old_eps, old_c, old_z = np.copy(u_theta), np.copy(u_zeta), np.copy(u_eps), np.copy(
        u_c), np.copy(u_z)

    for it in range(1000):
        # update z
        Elog_theta = Elog_dir(u_theta)
        B_tilde = np.exp(Elog_theta)
        Elog_eta = Elog_eta_k(u_eps)
        A_tilde = np.sum(u_c * np.exp(Elog_eta[..., np.newaxis]), axis=1)
        assert A_tilde.shape == (T0, T0)
        alpha_ls, beta_ls = list(), list()
        for d in range(seq_N):
            # shape of alpha and beta: (seq_L, T0)
            alpha_d, beta_d, u_zd = FB(A=A_tilde, B=B_tilde,
                                       seq=obs_arr[d, :obs_len[d]])
            u_z[d, :obs_len[d], :] = u_zd
            alpha_ls.append(alpha_d)
            beta_ls.append(beta_d)

        # update theta
        for w in range(obs_N):
            mask_arr = np.where(obs_arr == w)
            u_w = b0 + np.sum(u_z[mask_arr], axis=0)
            assert u_w.size == T0
            u_theta[:, w] = u_w

        # update zeta
        tmp_u_c = np.sum(u_c, axis=(0, 1))
        u_zeta[:, 0] = 1 + tmp_u_c
        arr_mask = 1 - np.tri(T0, T0, 0)
        u_zeta[:, 1] = alpha_0 + np.squeeze(arr_mask @ (tmp_u_c.reshape(-1, 1)))

        # calculate \xi_{d,i}(k,m)
        xi = np.zeros((seq_N, seq_L, T0, T1))
        for d in range(seq_N):
            for i in range(1, obs_len[d]):
                term_1 = alpha_ls[d][i - 1, :].reshape(-1, 1) * np.exp(Elog_eta)
                tmp = beta_ls[d][i] * np.exp(Elog_theta[:, obs_arr[d][i]])
                term_2 = np.ones((T0, T1))
                for k_ in range(T0):
                    term_2 *= (np.power(tmp[k_], u_c[:, :, k_]))
                # normalize for each i and d
                xi_di = term_1 * term_2 + 1e-100
                normalize_xi = np.sum(xi_di)
                xi_di /= normalize_xi
                xi[d, i, :, :] = xi_di

        # update epsilon
        tmp_xi_sum = np.sum(xi[:, 1:, :, :], axis=(0, 1))
        u_eps[:, :, 0] = 1 + tmp_xi_sum
        u_eps[:, :, 1] = tau_0
        for m_ in range(T1 - 1):
            u_eps[:, m_, 1] = np.sum(tmp_xi_sum[:, m_ + 1:], axis=1)

        # update c
        Elog_eta0 = Elog_eta_0(u_zeta)
        u_c = np.exp(Elog_eta0.reshape(1, 1, -1) + (tmp_xi_sum * Elog_eta).reshape(T0, T1, 1))

        diff_z = np.linalg.norm(old_z - u_z) / u_z.size
        diff_theta = np.linalg.norm(old_theta - u_theta) / u_theta.size
        diff_zeta = np.linalg.norm(old_zeta - u_zeta) / u_zeta.size
        diff_eps = np.linalg.norm(old_eps - u_eps) / u_eps.size
        diff_c = np.linalg.norm(old_c - u_c) / u_c.size

        if it % 20 == 0:
            print(f"iteration {it} : \n", f'change in theta is {diff_theta}')
            print(f'change in zeta is {diff_zeta}\n', f'change in eps is {diff_eps}')
            print(f'change in z is {diff_z}', '\n', f'change in c is {diff_c}')
            np.savez(f'../data/VI3-{it}-state-{num_state}_obs-{num_obs}_size-{num_seq}_maxL-{max_L}.npz', u_z=u_z)

        if diff_z < 1e-8:
            print('final diff_z: ', diff_z)
            np.savez(f'../data/VI3-final-{it}-state-{num_state}_obs-{num_obs}_size-{num_seq}_maxL-{max_L}.npz', u_z=u_z)
            break
        old_theta, old_zeta, old_eps, old_c, old_z = np.copy(u_theta), np.copy(u_zeta), np.copy(u_eps), np.copy(
            u_c), np.copy(u_z)


if __name__ == '__main__':
    loaded_npz = np.load(
        f"../data/hmm_syn_dataset(norefine_state-{num_state}_obs-{num_obs}_size-{num_seq}_maxL-{max_L}).npz",
        allow_pickle=True)
    observations = list(loaded_npz['observation'])
    # hidden_states = list(loaded_npz['real_hidden'])
    # noi_hidden_states = list(loaded_npz['noisy_hidden'])
    # transition_dist = np.vstack(loaded_npz['real_trans'])

    obs_len = [len(s) for s in observations]
    max_L = max(obs_len)

    # convert observation list to array, and pad with 0
    obs_arr = np.zeros((len(observations), max_L)) - 1
    for it in range(len(observations)):
        obs_arr[it, :obs_len[it]] = np.array(observations[it])
    obs_arr = obs_arr.astype(int)

    # truncate G to T0, the number of global hidden states, also the number of G_k
    T0 = 30
    # truncate G_k to T1
    T1 = 30
    VI(obs_arr=obs_arr, T0=T0, T1=T1, seq_N=num_seq, seq_L=max_L, obs_N=num_obs)
