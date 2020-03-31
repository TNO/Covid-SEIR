import numpy as np
from tqdm import tqdm

class InversionMethods:

    def __init__(self, ni=4, q=0.666667):

        self.ni = ni
        self.q = q

        if abs(q - 1.0) > 1e-10:
            alpha0 = (1.0 - q ** ni) / (q ** (ni - 1) - q ** ni)
        else:
            alpha0 = 1 / float(ni)

        self.alpha = alpha0 * q ** np.array(range(ni))

    def es_mda(self, forward, m_prior, y, sigma, *args_forward, **kwargs):

        """
        Perform ES_MDA smoother to estimate model parameters
        :param forward:         function that propagates array of parameter values into predictions
        :param m_prior:         list of prior parameter arrays
        :param y:               array with measured data
        :param sigma:           array with data error standard deviations (assumed independent)
        :param args_forward:    additional arguments required by forward
        :param kwargs:          additional keyword arguments required by forward
        :return:    MNew:       Smoothed ensemble of parameter arrays
                    GM_pre      Prior predictions, i.e. forward(m_prior)
                    GM_est      predictions with model estimate, i.e. forward(MNew)
        """

        MNew = m_prior[:]
        nI = self.ni  # number of MDA iterations
        alpha = self.alpha

        nM = len(m_prior[0])    # Nr of model parameter
        nE = len(m_prior)       # Nr of prior member
        nD = len(y)             # Nr of data points

        # data error covariance and inverse covariance matrices
        covD = np.square(sigma)
        covD = np.diag(covD)
        # Using pinv instead of inv for better performance in some cases
        covDInv = np.linalg.pinv(covD)

        M = []
        GM = []
        fw = []

        # Define mapping of model to observations
        map_model_to_obs = kwargs['G']
        output_index = kwargs['output_index']
        t_obs = kwargs['t_obs']

        # loop over the MDA iterations
        for iI in range(nI):

            # Run the model for every ensemble member and store predictions fw and simulated data GM
            GMiI = []
            fwiI = []
            for iE in tqdm(range(nE),desc='Running ensemble. Round {} of {}'.format(iI+1,nI)):
                fwiE = forward(MNew[iE],*args_forward)
                fwiI.append(fwiE)
                GMiI.append(map_model_to_obs(fwiE,t_obs,output_index))

            M.append(MNew)
            GM.append(GMiI)
            fw.append(fwiI)

            # only a forward run for the final iteration
            if iI == nI:
                break

            # inflate the data error covariance with the weight factor alpha for the current iteration
            cD_E = covD * alpha[iI]

            cdinv_E = covDInv / alpha[iI]
            sigdE = np.sqrt(cD_E)

            # create an nM x nE matrix with model parameters
            Mmatrix = np.asarray(MNew).reshape(nE, nM)
            Mmatrix = Mmatrix.transpose()

            # ensemble-mean model parameters
            Mmean = Mmatrix.mean(axis=1)

            # ensemble parameter anomalies
            Mt = np.zeros((nM, nE))
            for i in range(nE):
                Mt[:, i] = Mmean
            MP = Mmatrix - Mt

            # create an nD x nE matrix with the simulated data
            GMmatrix = np.asarray(GMiI).reshape(nE, nD)
            GMmatrix = GMmatrix.transpose()

            # ensemble-mean simulated data
            GMmean = GMmatrix.mean(axis=1)

            # ensemble simulated data anomalies
            Mt = np.zeros((nD, nE))
            for i in range(nE):
                Mt[:, i] = GMmean
            GMP = GMmatrix - Mt
            GMPT = GMP.transpose()

            # create random realizations of measurement noise consistent with the current alpha
            epsd = np.random.standard_normal(size=(nD, nE))
            for i in range(nD):
                epsd[i, :] = epsd[i, :] * sigdE[i, i]

            # create an ensemble of perturbed data
            data = np.asarray(y)
            DMC = np.zeros((nD, nE))
            for i in range(nE):
                DMC[:, i] = data + epsd[:, i]

            # construct and solve the ensemble update equation
            inn = (DMC - GMmatrix)
            if nD < nE:
                A = np.matmul(GMP, GMPT) + (nE - 1) * cD_E
                # solve A * W = inn for W
                W = np.linalg.solve(A, inn)
                sol = np.matmul(GMPT, W)
            else:
                A = np.matmul(cdinv_E, inn)
                A = np.matmul(GMPT, A)
                B = np.matmul(cdinv_E, GMP)
                B = np.matmul(GMPT, B)
                B = B + (nE - 1) * np.identity(nE)
                sol = np.linalg.solve(B, A)

            sol = np.matmul(MP, sol)
            MmatrixNew = Mmatrix + sol

            MNew = []
            for iE in range(nE):
                MNew.append(np.array(MmatrixNew)[:, iE])

        if 'print_results' in kwargs:
            if kwargs['print_results']:
                # Determine mean and standard deviation of prior and estimate
                mmean = []
                mstd = []
                GMmean = []
                GMstd = []
                for iI in range(len(M)):
                    # ensemble mean parameter estimates
                    mi = np.asarray(M[iI]).reshape(nE, nM)
                    mi = mi.transpose()
                    mm = mi.mean(axis=1)
                    mmean.append(mm)
                    # ensemble spread of parameter estimates
                    ms = mi.std(axis=1)
                    mstd.append(ms)
                    # ensemble mean data predictions
                    gmi = np.asarray(GM[iI]).reshape(nE, nD)
                    gmi = gmi.transpose()
                    gmm = gmi.mean(axis=1)
                    GMmean.append(gmm)
                    # ensemble spread of data predictions
                    gms = gmi.std(axis=1)
                    GMstd.append(gms)

                header = 'i[m] ||   m0_mean      m0_sig  '
                # for iI in range(1,len(M)):
                for iI in range(len(M) - 1, len(M)):  # add only last element...
                    header = header + '||   m%i_mean      m%i_sig  ' % (iI, iI)

                print(header)

                for i_m in range(nM):
                    # line = ' %2i  ' % i_m
                    line = args_forward[0]['free_param'][i_m]
                    # for iI in range(len(M)):
                    for iI in [0, len(M) - 1]:
                        mm = mmean[iI][i_m]
                        ms = mstd[iI][i_m]

                        line = line + '||  %7f    %7f  ' % (mm, ms)

                    print(line)

                print('')

                # determine prior and estimate for data predictions
                header = 'i[y]   y_meas  ||   y0_mean      y0_sig  '
                # for iI in range(1,len(M)):
                for iI in range(len(M) - 1, len(M)):  # add only last element...
                    header = header + '||   y%i_mean      y%i_sig  ' % (iI, iI)

                print(header)

                for i_d in range(nD):
                    line = ' %2i  ' % i_d
                    line = line + '%7f  ' % y[i_d]
                    # for iI in range(len(M)):
                    for iI in [0, len(M) - 1]:
                        mm = GMmean[iI][i_d]
                        ms = GMstd[iI][i_d]

                        line = line + '||  %7f    %7f  ' % (mm, ms)

                    print(line)

                print('')

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        return {'M': M, 'GM': GM, 'fw': fw}


def model(m):
    y0 = m[0] ** 2 + m[1] + 0.5* m[2] ** 1.2
    y1 = y0 + m[0] * m[1]
    y = [y0, y1]

    return y


# def main():
#     im = InversionMethods(ni=4)
#
#     forward = model
#
#     np.random.seed(12345)
#
#     print("testing the ES-MDA code")
#     m_prior = []
#     for i in range(10):
#         m1 = np.random.normal(0.7, 0.1, 1)
#         m2 = np.random.normal(0.3, 0.1, 1)
#         m3 = np.random.normal(1.0, 0.4, 1)
#         mr = np.array([m1, m2, m3])
#         m_prior.append(mr)
#
#     Mtrue = [0.6, 0.2, 0.8]
#     y_obs = forward(Mtrue)
#     print('data value: ' + str(y_obs))
#     sigma = 0.01 * np.ones((len(y_obs)))
#
#     add_arg = []
#     kwargs = {'nplots': 0,
#               'print_results': 1}
#
#     results = im.es_mda(forward, m_prior, y_obs, sigma, *add_arg, **kwargs)
#     M = results['M']
#     GM = results['GM']
#     fw = results['fw']
#
#
# if __name__ == "__main__":
#     main()
