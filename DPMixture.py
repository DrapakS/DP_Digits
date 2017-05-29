import numpy as np
from scipy.special import digamma as psi
from scipy.special import gammaln
from scipy.special import gamma as G
import matplotlib.pyplot as plt
__author__ = 'Stepan'


class DPMixture:
    def __init__(self, X, alpha, a, b):
        self.X = X
        self.alpha = alpha
        self.a = a
        self.b = b
        c = 10
        self.N = X.shape[0]
        self.D = X.shape[1]
        self.T = int(round(c * alpha * np.log(1 + self.N/alpha)))
        gamma = np.random.uniform(size=(self.N, self.T))
        self.gamma = (gamma.T/np.sum(gamma, axis=1)).T

    def var_inference(self, num_start=1, display=True, max_iter=100, tol_L=1e-4):
        best_L = -np.inf
        for i in range(0, num_start):
            if i != 0:
                gamma = np.random.uniform(size=(self.N, self.T))
                self.gamma = (gamma.T/np.sum(gamma, axis=1)).T
            l_list = []
            for j in range(0, max_iter):
                av = 1 + np.sum(self.gamma, axis=0)
                t_bv = np.sum(self.gamma) + self.alpha
                bv = np.repeat(t_bv, self.T)
                gamma_colsum = av - 1
                gamma_colcumsum = np.cumsum(gamma_colsum)
                bv -= gamma_colcumsum

                atheta = self.a + np.dot((self.gamma).T, self.X)
                btheta = self.b + np.dot((self.gamma).T, (1 - self.X))

                log_theta_expct = psi(atheta) - psi(atheta + btheta)
                log_neg_theta_expct = psi(btheta) - psi(atheta + btheta)
                log_v_expct = psi(av) - psi(av + bv)
                log_neg_v_expct = psi(bv) - psi(av + bv)
                s1 = np.dot(log_theta_expct, self.X.T)
                s2 = np.dot(log_neg_theta_expct, (1 - self.X).T)
                s3 = log_v_expct
                s4 = np.cumsum(log_neg_v_expct) - log_neg_v_expct

                gamma = np.exp(s1.T + s2.T + s3 + s4)
                self.gamma = (gamma.T/np.sum(gamma, axis=1)).T

                fsz = self.T * self.D
                lf_s1 = np.sum((self.a - 1) * log_theta_expct + (self.b - 1) * log_neg_theta_expct) + \
                            fsz * (gammaln(self.a + self.b) + G(self.a) + G(self.b))
                lf_s2 = np.sum((self.alpha - 1) * log_neg_v_expct) + \
                            self.T * (gammaln(self.alpha + 1) - gammaln(self.alpha) - gammaln(1))
                lf_s3 = np.sum((s1 + s2) * self.gamma.T)
                lf_s4 = np.sum((s3 + s4) * self.gamma)
                l_full = lf_s1 + lf_s2 + lf_s3 + lf_s4

                lv_s1 = np.sum((av - 1) * (psi(av) - psi(av + bv)))
                lv_s2 = np.sum((bv - 1) * (psi(bv) - psi(av + bv)))
                lv_s3 = np.sum(gammaln(av + bv) - gammaln(av) - gammaln(bv))
                l_v = lv_s1 + lv_s2 + lv_s3

                ltheta_s1 = np.sum((atheta - 1) * (psi(atheta) - psi(atheta + btheta)))
                ltheta_s2 = np.sum((btheta - 1) * (psi(btheta) - psi(atheta + btheta)))
                ltheta_s3 = np.sum(gammaln(atheta + btheta) - gammaln(atheta) - gammaln(btheta))
                l_theta = ltheta_s1 + ltheta_s2 + ltheta_s3

                l_z = np.sum(self.gamma * np.log(self.gamma))

                l_final = l_full - l_v - l_theta - l_z

                if display:
                    print("Iteration_number=" + str(j))
                    print("L = " + str(l_final))
                    cl_num = np.argmax(self.gamma, axis=1)
                    clustersNums = np.unique(cl_num)
                    n_cl = clustersNums.shape[0]
                    print("number of components:" + str(n_cl))
                    print('=' * 20)
                l_list.append(l_final)
                if j != 0:
                    if abs(l_list[j] - l_list[j - 1]) < tol_L:
                        break
            if l_list[-1] > best_L:
                #print("!!!")
                best_L = l_list[-1]
                best_gamma = self.gamma
                best_L_list = l_list

            self.gamma = best_gamma
        return best_L_list

    def show_clusters(self):
        X = np.array(self.X, dtype=float)
        s_gamma = self.gamma
        cl_num = np.argmax(s_gamma, axis=1)
        clustersNums = np.unique(cl_num)
        n_cl = clustersNums.shape[0]

        f, axarr = plt.subplots(2, (n_cl % 2) + n_cl/2)
        print(clustersNums.shape[0])
        for i in range(len(clustersNums)):
            w_cl = cl_num == clustersNums[i]
            meanIm = np.sum(X[w_cl, :], axis=0)/np.sum(w_cl)
            sz = int(np.sqrt(meanIm.shape[0]))
            meanIm = np.reshape(meanIm, (sz, sz))
            axarr[i%2, i/2].imshow(meanIm, cmap='gray')

        for i in range(n_cl + n_cl % 2):
            axarr[i%2, i/2].axis('off')

        plt.show()

    def add_samples(self, X):
        self.X = np.vstack((self.X, X))
        gamma = np.random.uniform(size=(X.shape[0], self.T))
        gamma = (gamma.T/np.sum(gamma, axis=1)).T
        self.gamma = np.vstack((self.gamma, gamma))
        #print(self.X.shape, self.gamma.shape)
        return self

