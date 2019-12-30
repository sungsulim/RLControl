import numpy as np
import scipy as sp
import scipy.stats
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


class GaussianMixture1D:
    def __init__(self, mixture_probs, means, stds):
        self.num_mixtures = len(mixture_probs)
        self.mixture_probs = mixture_probs
        self.means = means
        self.stds = stds

    def sample(self, num_samples=1):
        mixture_ids = np.random.choice(self.num_mixtures, size=num_samples, p=self.mixture_probs)
        result = np.zeros([num_samples])
        for sample_idx in range(num_samples):
            result[sample_idx] = np.random.normal(
                loc=self.means[mixture_ids[sample_idx]],
                scale=self.stds[mixture_ids[sample_idx]]
            )
        return result

    def logpdf(self, samples):
        mixture_logpdfs = np.zeros([len(samples), self.num_mixtures])
        for mixture_idx in range(self.num_mixtures):

            mixture_logpdfs[:, mixture_idx] = scipy.stats.norm.logpdf(
                samples,
                loc=self.means[mixture_idx],
                scale=self.stds[mixture_idx]
            )
        return sp.special.logsumexp(mixture_logpdfs + np.log(self.mixture_probs), axis=1)

    def pdf(self, samples):
        return np.exp(self.logpdf(samples))


def approx_kl(gmm_1, gmm_2, xs):
    ys = gmm_1.pdf(xs) * (gmm_1.logpdf(xs) - gmm_2.logpdf(xs))
    return np.trapz(ys, xs)


def minimize_pq(p, xs, q_means, q_stds):
    q_mean_best = None
    q_std_best = None
    kl_best = np.inf
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            kl = approx_kl(p, q, xs)
            if kl < kl_best:
                kl_best = kl
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, kl_best

def minimize_pq_cem(p, xs, q_mean_params_init, q_std_params_init, q_mean_bounds, q_std_bounds, num_iter):

    num_samples = 1000
    rho = 0.01

    q_mean_best = None
    q_std_best = None
    kl_best = np.inf

    q_mean_params = q_mean_params_init
    q_std_params = q_std_params_init

    for i in range(num_iter):

        # create distribution and sample
        q_mean_dist = GaussianMixture1D(np.array([1]), np.array([q_mean_params[0]]), np.array([q_mean_params[1]]))
        q_std_dist = GaussianMixture1D(np.array([1]), np.array([q_std_params[0]]), np.array([q_std_params[1]]))

        sampled_mean_arr = np.array(q_mean_dist.sample(num_samples=30))
        sampled_std_arr = np.array(q_std_dist.sample(num_samples=30))

        # sample candidate distribution
        sampled_q_arr = [GaussianMixture1D(np.array([1]), np.array([sampled_mean]), np.array([sampled_std])) for sampled_mean, sampled_std in zip(sampled_mean_arr, sampled_std_arr)]
        # compute kl
        sampled_kl_arr = np.array([approx_kl(p, sampled_q, xs) for sampled_q in sampled_q_arr])
        print("avg KL(p||q) at iter {}: {}".format(i, np.mean(sampled_kl_arr)))

        selected_idxs = np.argsort(sampled_kl_arr)[:int(num_samples * rho)]

        # update params
        new_q_mean_dist = GaussianMixture(n_components=1, covariance_type='diag').fit(sampled_mean_arr[selected_idxs].reshape(-1,1))
        new_q_std_dist = GaussianMixture(n_components=1, covariance_type='diag').fit(sampled_std_arr[selected_idxs].reshape(-1,1))

        q_mean_params = [new_q_mean_dist.means_[0][0], new_q_mean_dist.covariances_[0][0]]
        q_std_params = [new_q_std_dist.means_[0][0], new_q_std_dist.covariances_[0][0]]

    final_q = GaussianMixture1D(np.array([1]), np.array([q_mean_params[0]]), np.array([q_std_params[0]]))
    final_kl = approx_kl(p, final_q, xs)
    return final_q, final_kl


def minimize_qp_cem(p, xs, q_mean_params_init, q_std_params_init, q_mean_bounds, q_std_bounds, num_iter):
    num_samples = 1000
    rho = 0.01

    q_mean_best = None
    q_std_best = None
    kl_best = np.inf

    q_mean_params = q_mean_params_init
    q_std_params = q_std_params_init

    for i in range(num_iter):
        # create distribution and sample
        q_mean_dist = GaussianMixture1D(np.array([1]), np.array([q_mean_params[0]]), np.array([q_mean_params[1]]))
        q_std_dist = GaussianMixture1D(np.array([1]), np.array([q_std_params[0]]), np.array([q_std_params[1]]))

        sampled_mean_arr = np.array(q_mean_dist.sample(num_samples=30))
        sampled_std_arr = np.array(q_std_dist.sample(num_samples=30))

        # sample candidate distribution
        sampled_q_arr = [GaussianMixture1D(np.array([1]), np.array([sampled_mean]), np.array([sampled_std])) for
                         sampled_mean, sampled_std in zip(sampled_mean_arr, sampled_std_arr)]
        # compute kl
        sampled_kl_arr = [approx_kl(sampled_q, p, xs) for sampled_q in sampled_q_arr]
        print("avg KL(q||p) at iter {}: {}".format(i, np.mean(sampled_kl_arr)))

        selected_idxs = np.argsort(sampled_kl_arr)[:int(num_samples * rho)]

        # update params
        new_q_mean_dist = GaussianMixture(n_components=1, covariance_type='diag').fit(sampled_mean_arr[selected_idxs].reshape(-1,1))
        new_q_std_dist = GaussianMixture(n_components=1, covariance_type='diag').fit(sampled_std_arr[selected_idxs].reshape(-1,1))

        q_mean_params = [new_q_mean_dist.means_[0][0], new_q_mean_dist.covariances_[0][0]]
        q_std_params = [new_q_std_dist.means_[0][0], new_q_std_dist.covariances_[0][0]]

    final_q = GaussianMixture1D(np.array([1]), np.array([q_mean_params[0]]), np.array([q_std_params[0]]))
    final_kl = approx_kl(final_q, p, xs)
    return final_q, final_kl

def minimize_qp(p, xs, q_means, q_stds):
    q_mean_best = None
    q_std_best = None
    kl_best = np.inf
    for q_mean in q_means:
        for q_std in q_stds:
            q = GaussianMixture1D(np.array([1]), np.array([q_mean]), np.array([q_std]))
            kl = approx_kl(q, p, xs)
            if kl < kl_best:
                kl_best = kl
                q_mean_best = q_mean
                q_std_best = q_std

    q_best = GaussianMixture1D(np.array([1]), np.array([q_mean_best]), np.array([q_std_best]))
    return q_best, kl_best


def main():
    # configuration
    distance_min = 0
    distance_max = 3
    num_points = 5

    save_file = False

    p_mixture_probs = np.array([0.5, 0.5])
    p_stds = np.array([0.2, 0.2])
    distance_list = np.linspace(distance_min, distance_max, num_points)

    q_stds_min = 0.0001
    q_stds_max = 7

    # Exhaustive search
    num_q_means = 100
    num_q_stds = 100

    # CEM search
    q_mean_mu = 0.0
    q_mean_sigma = 1.0

    q_std_mu = 0.1
    q_std_sigma = 1.0

    cem_iter = 50

    p = [None] * num_points
    q_best_forward = [None] * num_points
    kl_best_forward = [None] * num_points
    q_best_reverse = [None] * num_points
    kl_best_reverse = [None] * num_points

    trapz_xs_arr = [None] * num_points

    for idx, dist in enumerate(distance_list):

        p_means = np.array([-dist/2, dist/2])

        p[idx] = GaussianMixture1D(p_mixture_probs, p_means, p_stds)

        q_means_min = np.min(p_means) - 1
        q_means_max = np.max(p_means) + 1

        # for exhaustive search
        # q_means = np.linspace(q_means_min, q_means_max, num_q_means)
        # q_stds = np.linspace(q_stds_min, q_stds_max, num_q_stds)

        # for cem search

        # get wide range of sample points
        trapz_xs_min = np.min(np.append(p_means, q_means_min)) - 3 * np.max(np.append(p_stds, q_stds_max))
        trapz_xs_max = np.max(np.append(p_means, q_means_min)) + 3 * np.max(np.append(p_stds, q_stds_max))

        num_trapz_points = 1000
        trapz_xs = np.linspace(trapz_xs_min, trapz_xs_max, num_trapz_points)

        trapz_xs_arr[idx] = trapz_xs

        # exhaustive search
        # q_best_forward[idx], kl_best_forward[idx] = minimize_pq(
        #     p[idx], trapz_xs, q_means, q_stds
        # )
        # q_best_reverse[idx], kl_best_reverse[idx] = minimize_qp(
        #     p[idx], trapz_xs, q_means, q_stds
        # )

        # cem search
        print('** forward KL **')
        q_best_forward[idx], kl_best_forward[idx] = minimize_pq_cem(p[idx], trapz_xs, [q_mean_mu, q_mean_sigma], [q_std_mu, q_std_sigma], [q_means_min, q_means_max], [q_stds_min, q_stds_max], cem_iter)
        print('** reverse KL **')
        q_best_reverse[idx], kl_best_reverse[idx] = minimize_qp_cem(p[idx], trapz_xs, [q_mean_mu, q_mean_sigma], [q_std_mu, q_std_sigma], [q_means_min, q_means_max], [q_stds_min, q_stds_max], cem_iter)
        print("")

        # q_best_reverse[idx], kl_best_reverse[idx] = minimize_qp_cem(p[idx], trapz_xs, q_mean_init, q_std_init, q_means_min, q_means_max, q_stds_min, q_stds_max, cem_iter)

    # plotting
    fig, axs = plt.subplots(nrows=1, ncols=num_points, sharex=True, sharey=True)
    # fig.set_size_inches(8, 1.5)
    # plt.figure(figsize=(20, 6))

    for idx, dist in enumerate(distance_list):
        xs_min = -distance_max/2 - 1
        xs_max = distance_max/2 + 1
        num_plot_points = 1000
        xs = np.linspace(xs_min, xs_max, num_plot_points)
        axs[idx].plot(xs, p[idx].pdf(xs), label='$p$', color='black')
        axs[idx].plot(xs, q_best_forward[idx].pdf(xs), label='$\mathrm{argmin}_q \,\mathrm{KL}(p || q)$', color='black', linestyle='dashed')
        axs[idx].plot(xs, q_best_reverse[idx].pdf(xs), label='$\mathrm{argmin}_q \,\mathrm{KL}(q || p)$', color='blue', linestyle='dotted')

        # log info
        print("*** mode at : {}, \nkl(p||q): mean {}, std {}, kl_val {},"
              "\nkl(q||p): mean {}, std {}, kl_val {}".format([-dist/2, dist/2], q_best_forward[idx].means, q_best_forward[idx].stds, kl_best_forward[idx], q_best_reverse[idx].means, q_best_reverse[idx].stds, kl_best_reverse[idx]))

        optimal_mean = dist/2
        optimal_std = q_stds_min
        optimal_q = GaussianMixture1D(np.array([1]), np.array([optimal_mean]), np.array([optimal_std]))
        # optimal_kl_forward_val = approx_kl(p[idx], optimal_q, trapz_xs_arr[idx])
        optimal_kl_reverse_val = approx_kl(optimal_q, p[idx], trapz_xs_arr[idx])

        # print('optimal kl(p||q): mean {}, std {}, kl_val {}'.format(optimal_mean, optimal_std, optimal_kl_forward_val))
        print('optimal kl(q||p): mean {}, std {}, kl_val {}\n'.format(optimal_mean, optimal_std, optimal_kl_reverse_val))

        axs[idx].spines['right'].set_visible(False)
        axs[idx].spines['top'].set_visible(False)
        # axs[idx].set_yticks([])
        # axs[idx].set_xticks([])

        axs[idx].set_title('dist: {}'.format(dist))

    axs[2].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.05), fontsize='small')
    filenames = ['reverse_forward_kl.pdf', 'reverse_forward_kl.png']

    if save_file:
        for filename in filenames:
            fig.savefig(filename, dpi=200)
            print('Saved to {}'.format(filename))

    plt.suptitle("KL CEM")
    plt.show()

if __name__ == '__main__':
    main()
