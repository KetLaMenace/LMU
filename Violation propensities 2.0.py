import numpy as np
import matplotlib.pyplot as plt

tau = 2
n = 4
ph = .55
pl = .45
u = 1.25
d = .8

w = 1.1
beta = .9
lam = 3
delta = .5
alpha = 0.5
alphaP = 1
eta = 1

#probability tensor with numpy and brains, still partly unexplained
#index [k,l,kk,ll] means k (kk) ups for the good asset at time tau (between tau and tau'), and l (ll) for the bad asset
tau_list = np.arange(tau + 1)
n_list = np.arange(n + 1)
L, K, KK, LL = np.meshgrid(tau_list, tau_list, n_list, n_list)
Del, DelP = K - L, K + KK - L - LL
#UNEXPLAINED : when meshing into a dimension>2, the first 2 indices appear to be switched
P = np.ones((tau + 1, tau + 1, n + 1, n + 1))
for i in tau_list[1:]:
    P[i, :, :, :] = (P[i - 1, :, :, :] * (tau - i + 1)) / i
    P[:, i, :, :] = (P[:, i - 1, :, :] * (tau - i + 1)) / i
for i in n_list[1:]:
    P[:, :, i, :] = (P[:, :, i - 1, :] * (n - i + 1)) / i
    P[:, :, :, i] = (P[:, :, :, i - 1] * (n - i + 1)) / i
P = P * (ph ** (K + KK) * (1 - ph) ** (tau + n - K - KK) * pl ** (L + LL) * (1 - pl) ** (tau + n - L - LL))


#relative wealth variation after t time increments with i price increases
def wealth(t, ups):
    return u ** ups * d ** (t - ups)


W_vector = wealth(n, n_list)
L, K, Wh, Wl = np.meshgrid(tau_list, tau_list, W_vector, W_vector)
#Wh is the relative wealth variation at tau' after investing in the good asset
E_Wh = (P * Wh).sum()
E_Wl = (P * Wl).sum()
Wlottery = (Wh + Wl) / 2
E_lottery = (E_Wh + E_Wl) / 2
E_optimal_play = w * (((Del > 0) * Wh * P).sum() + ((Del < 0) * Wl * P).sum() + ((Del == 0) * Wlottery * P).sum()) * (((DelP > 0) * P).sum() * E_Wh + ((DelP < 0) * P).sum() * E_Wl + ((DelP == 0) * P).sum() * E_lottery)


def g_get():
    g = 0
    while wealth(n, g) < 1:
        g += 1
    return g


g = g_get()
print('g = '+str(g))

#getting theta from Q^AO aka "thetifying" the q
def thetify(q, pa, pb):
    infinity = max(tau + 1, n + 1)
    #we send large thetas to infinity and negative ones to 0 so that the investor behavior is consistent
    return (q > .5) * (q < 1) * np.ceil(np.log(abs(q / (1 - q))) / np.log(pa * (1 - pb) / pb / (1 - pa))) + (q <= .5) * 0 + (q >= 1) * infinity


#expected value of a random variable taking value L[i] with probability binom(n,i)p^i(1-p)^(n-i)
def E(L, p):  ##L is the array of values X, or an array of line arrays Xi
    length_X = L.shape[-1]
    if len(L.shape) == 1 : L = L[np.newaxis, :]
    length = L.shape[0]
    sum = np.zeros((length))
    binom = 1
    for ups in range(length_X) :
        sum += L[:, ups] * binom * p ** ups * (1 - p) ** (length_X - 1 - ups)
        binom = (binom * (length_X - 1 - ups)) / (ups + 1)
    sum = sum
    return sum  ##return is the array [E(Xi)]

print(E(np.ones(500),.1))

#PT utility
def v(y, beta, lam):
    return np.abs(y) ** beta * (-lam * (y < 0) + (y > 0))
#As v is self-similar, we could ditch the variable w completely. We choose to keep it so that more general utility forms can be added easily, and set w=1 in all computations


def QAOtau(beta, lam, w, r):
    return (v(w - r, beta, lam) - E(v(w * W_vector - r, beta, lam), pl)) / (E(v(w * W_vector - r, beta, lam), ph) - E(v(w * W_vector - r, beta, lam), pl))


def QAOtauP(beta, lam, w, c, r):
    Utility_tensor = v(np.outer(w + c, W_vector) - r, beta, lam)
    return (v(w + c - r, beta, lam) - E(Utility_tensor, pl)) / (E(Utility_tensor, ph) - E(Utility_tensor, pl))

#PT and RU

def QAOtauP_RU(beta, lam, delta, w, c):
    Utility_tensor = v(np.outer(w + c, W_vector) - w, beta, lam)
    return (v(c, beta, lam)/delta - E(Utility_tensor, pl)) / (E(Utility_tensor, ph) - E(Utility_tensor, pl))

def propensities(beta, lam, delta, alpha, alphaP, eta, w, theory) :
    r=0
    if theory == 'BRN' :
        q = QAOtau(beta, lam, w, 0)
        odds = (q/(1-q))**alphaP
        theta = thetify(odds/(1+odds), ph, pl)
        if theta == 0: return 'Boundary solution'
        KA, KB = (DelP + (alpha - 1) * Del) * alphaP >= theta, -(DelP + (alpha - 1) * Del) * alphaP >= theta
        SA, SB = KB, KA
    elif theory == 'EE' :
        theta = thetify(QAOtau(beta, lam, w, 0), ph, pl)
        if theta == 0: return 'Boundary solution'
        KA, KB = 2 * ((1 - eta) * Del + eta * (DelP - Del)) >= theta, -(2 * ((1 - eta) * Del + eta * (DelP - Del))) >= theta
        SA, SB = KB, KA
    elif theory == 'BMR' :
        theta = thetify(QAOtau(beta, lam, w, 0), ph, pl)
        if theta == 0: return 'Boundary solution'
        KA, KB = -DelP >= theta, DelP >= theta
        SA, SB = KB, KA
    elif theory == 'LEPT' :
        r = ref_point(beta,lam,w)
        theta = thetify(QAOtau(beta, lam, w, r), ph, pl)
        if theta == 0:
            print('r='+str(r))
            return 'Boundary solution'
        ThetaP_vector = thetify(QAOtauP(beta, lam, w, w * (W_vector - 1), r), ph, pl)
        print('PT theta prime = ' + str(ThetaP_vector))
        K, K, ThetaPh, ThetaPl = np.meshgrid(tau_list, tau_list, ThetaP_vector, ThetaP_vector)
        # investment decision at tau'
        KA, KB = DelP >= ThetaPh, -DelP >= ThetaPl
        SA, SB = (-DelP >= ThetaPh) * (ThetaPh > 0) + (-DelP > 0) * (ThetaPh == 0), (DelP >= ThetaPl) * (ThetaPl > 0) + (DelP > 0) * (ThetaPl == 0)
    else :
        theta = thetify(QAOtau(beta, lam, w, w), ph, pl)
        if theta == 0 : return 'Boundary solution'
        if theory == 'RU' :
            ThetaPK_vector = thetify(QAOtauP_RU(beta, lam, delta, w, w * (W_vector - 1)), ph, pl)
            print('RU theta prime = ' + str(ThetaPK_vector))
            K, K, ThetaPhK, ThetaPlK = np.meshgrid(tau_list, tau_list, ThetaPK_vector, ThetaPK_vector)
            #investment decision at tau'
            KA, KB = DelP >= ThetaPhK, -DelP >= ThetaPlK
            SA, SB = -DelP >= theta, DelP >= theta
        elif theory == 'PT' :
            ThetaP_vector = thetify(QAOtauP(beta, lam, w, w * (W_vector - 1), w), ph, pl)
            print('PT theta prime = ' + str(ThetaP_vector))
            K, K, ThetaPh, ThetaPl = np.meshgrid(tau_list, tau_list, ThetaP_vector, ThetaP_vector)
            # investment decision at tau'
            KA, KB = DelP >= ThetaPh, -DelP >= ThetaPl
            SA, SB = (-DelP >= ThetaPh) * (ThetaPh > 0) + (-DelP > 0) * (ThetaPh == 0), (DelP >= ThetaPl) * (ThetaPl > 0) + (DelP > 0) * (ThetaPl == 0)
        else : return None
    QA, QB = 1 - KA - SA, 1 - KB - SB
    # investment decision at tau
    A, B = (Del >= theta), (Del <= -theta)
    if theory == 'BMR' :
        temp = A.copy()
        A, B = B.copy(), temp
    #gains or losses
    GA, GB = KK >= g, LL >= g  ##G|A,B
    #benchmark event
    S1A, S1B = DelP >= 0, DelP <= 0  ##1st order violation to switch conditional on A, on B   1-x to get keep violations
    Q2 = np.abs(DelP) >= theta  ##2nd order to liquidate |A,B   1-x to get other one
    #propensities
    pG = ((P * A * GA).sum() + (P * B * GB).sum())
    pL = ((P * A * (1-GA)).sum() + (P * B * (1-GB)).sum())
    pGSK = ((P * GA * A * S1A * (1 - Q2)).sum() + (P * GB * B * S1B * (1 - Q2)).sum()) / pG
    pLSK = ((P * (1 - GA) * A * S1A * (1 - Q2)).sum() + (P * (1 - GB) * B * S1B * (1 - Q2)).sum()) / pL
    kSKG = ((P * A * GA * S1A * (1 - Q2) * KA).sum() + (P * B * GB * S1B * (1 - Q2) * KB).sum()) / ((P * A * GA * S1A * (1 - Q2)).sum() + (P * B * GB * S1B * (1 - Q2)).sum())
    kSKL = ((P * A * (1 - GA) * S1A * (1 - Q2) * KA).sum() + (P * B * (1 - GB) * S1B * (1 - Q2) * KB).sum()) / ((P * A * (1 - GA) * S1A * (1 - Q2)).sum() + (P * B * (1 - GB) * S1B * (1 - Q2)).sum())
    sSKG = ((P * A * GA * S1A * (1 - Q2) * SA).sum() + (P * B * GB * S1B * (1 - Q2) * SB).sum()) / ((P * A * GA * S1A * (1 - Q2)).sum() + (P * B * GB * S1B * (1 - Q2)).sum())
    sSKL = ((P * A * (1 - GA) * S1A * (1 - Q2) * SA).sum() + (P * B * (1 - GB) * S1B * (1 - Q2) * SB).sum()) / ((P * A * (1 - GA) * S1A * (1 - Q2)).sum() + (P * B * (1 - GB) * S1B * (1 - Q2)).sum())
    pGKS = ((P * GA * A * (1 - S1A) * (1 - Q2)).sum() + (P * GB * B * (1 - S1B) * (1 - Q2)).sum()) / pG
    pLKS = ((P * (1 - GA) * A * (1 - S1A) * (1 - Q2)).sum() + (P * (1 - GB) * B * (1 - S1B) * (1 - Q2)).sum()) / pL
    sKSG = ((P * A * GA * (1 - S1A) * (1 - Q2) * (1 - KA - QA)).sum() + (P * B * GB * (1 - S1B) * (1 - Q2) * (1 - KB - QB)).sum()) / ((P * A * GA * (1 - S1A) * (1 - Q2)).sum() + (P * B * GB * (1 - S1B) * (1 - Q2)).sum())
    sKSL = ((P * A * (1 - GA) * (1 - S1A) * (1 - Q2) * (1 - KA - QA)).sum() + (P * B * (1 - GB) * (1 - S1B) * (1 - Q2) * (1 - KB - QB)).sum()) / ((P * A * (1 - GA) * (1 - S1A) * (1 - Q2)).sum() + (P * B * (1 - GB) * (1 - S1B) * (1 - Q2)).sum())
    kKSG = ((P * A * GA * (1 - S1A) * (1 - Q2) * KA).sum() + (P * B * GB * (1 - S1B) * (1 - Q2) * KB).sum()) / ((P * A * GA * (1 - S1A) * (1 - Q2)).sum() + (P * B * GB * (1 - S1B) * (1 - Q2)).sum())
    kKSL = ((P * A * (1 - GA) * (1 - S1A) * (1 - Q2) * KA).sum() + (P * B * (1 - GB) * (1 - S1B) * (1 - Q2) * KB).sum()) / ((P * A * (1 - GA) * (1 - S1A) * (1 - Q2)).sum() + (P * B * (1 - GB) * (1 - S1B) * (1 - Q2)).sum())
    pGKQ = ((P * GA * A * (1 - S1A) * Q2).sum() + (P * GB * B * (1 - S1B) * Q2).sum()) / pG
    pLKQ = ((P * (1 - GA) * A * (1 - S1A) * Q2).sum() + (P * (1 - GB) * B * (1 - S1B) * Q2).sum()) / pL
    lKQG = ((P * A * GA * (1 - S1A) * Q2 * QA).sum() + (P * B * GB * (1 - S1B) * Q2 * QB).sum()) / ((P * A * GA * (1 - S1A) * Q2).sum() + (P * B * GB * (1 - S1B) * Q2).sum())
    lKQL = ((P * A * (1 - GA) * (1 - S1A) * Q2 * QA).sum() + (P * B * (1 - GB) * (1 - S1B) * Q2 * QB).sum()) / ((P * A * (1 - GA) * (1 - S1A) * Q2).sum() + (P * B * (1 - GB) * (1 - S1B) * Q2).sum())
    kKQG = ((P * A * GA * (1 - S1A) * Q2 * KA).sum() + (P * B * GB * (1 - S1B) * Q2 * KB).sum()) / ((P * A * GA * (1 - S1A) * Q2).sum() + (P * B * GB * (1 - S1B) * Q2).sum())
    kKQL = ((P * A * (1 - GA) * (1 - S1A) * Q2 * KA).sum() + (P * B * (1 - GB) * (1 - S1B) * Q2 * KB).sum()) / ((P * A * (1 - GA) * (1 - S1A) * Q2).sum() + (P * B * (1 - GB) * (1 - S1B) * Q2).sum())
    pGSQ = ((P * GA * A * S1A * Q2).sum() + (P * GB * B * S1B * Q2).sum()) / pG
    pLSQ = ((P * (1 - GA) * A * S1A * Q2).sum() + (P * (1 - GB) * B * S1B * Q2).sum()) / pL
    lSQG = ((P * A * GA * S1A * Q2 * QA).sum() + (P * B * GB * S1B * Q2 * QB).sum()) / ((P * A * GA * S1A * Q2).sum() + (P * B * GB * S1B * Q2).sum())
    lSQL = ((P * A * (1 - GA) * S1A * Q2 * QA).sum() + (P * B * (1 - GB) * S1B * Q2 * QB).sum()) / ((P * A * (1 - GA) * S1A * Q2).sum() + (P * B * (1 - GB) * S1B * Q2).sum())
    sSQG = ((P * A * GA * S1A * Q2 * SA).sum() + (P * B * GB * S1B * Q2 * SB).sum()) / ((P * A * GA * S1A * Q2).sum() + (P * B * GB * S1B * Q2).sum())
    sSQL = ((P * A * (1 - GA) * S1A * Q2 * SA).sum() + (P * B * (1 - GB) * S1B * Q2 * SB).sum()) / ((P * A * (1 - GA) * S1A * Q2).sum() + (P * B * (1 - GB) * S1B * Q2).sum())
    kG = ((P * A * GA * (1 - S1A * Q2) * KA).sum() + (P * B * GB * (1 - S1B * Q2) * KB).sum()) / pG
    kL = ((P * A * (1 - GA) * (1 - S1A * Q2) * KA).sum() + (P * B * (1 - GB) * (1 - S1B * Q2) * KB).sum()) / pL
    PGR, PLR = pGSK * (1 - np.nan_to_num(kSKG)) + pGKS * (1 - np.nan_to_num(kKSG)) + pGKQ * (1 - np.nan_to_num(kKQG)) + pGSQ * np.nan_to_num(lSQG + sSQG), pLSK * (1 - np.nan_to_num(kSKL)) + pLKS * (1 - np.nan_to_num(kKSL)) + pLKQ * (1 - np.nan_to_num(kKQL)) + pLSQ * np.nan_to_num(lSQL + sSQL)
    return kSKG, sKSG, lKQG, lSQG, kSKL, sKSL, lKQL, lSQL, sSKG, kKSG, kKQG, sSQG, sSKL, kKSL, kKQL, sSQL, kG, kL, PGR, PLR, theta, r

def propensities_disp(beta,lam,delta,alpha,alphaP,eta,w, theory) :
    list = propensities(beta,lam,delta,alpha,alphaP,eta,w,theory)
    plt.figure()
    plt.axis('off')
    if theory == 'PT' : plt.title('Status-quo prospect theory', fontsize='xx-large')
    elif theory == 'RU' : plt.title('Realization utility', fontsize='xx-large')
    elif theory == 'BRN' : plt.title('Base-rate neglect', fontsize='xx-large')
    elif theory == 'EE' : plt.title('Extrapolative expectations', fontsize='xx-large')
    elif theory == 'BMR' : plt.title('Belief in mean reversion', fontsize='xx-large')
    elif theory == 'LEPT' : plt.title('Lagged-expectations Prospect theory', fontsize='xx-large')
    if type(list) == str :
        plt.text(.5,.5,r'Agent always invests at time $\tau$', horizontalalignment='center', fontsize='x-large')
    elif list[:-1] == np.nan :
        plt.text(.5,.5,r'Agent never invests at time $\tau$', horizontalalignment='center', fontsize='x-large')
    else :
        plt.text(.18,.82,r'$\kappa_{SK}^{G} = $'+str(list[0])+'\n'r'$\sigma_{KS}^{G} = $'+str(list[1])+'\n'+r'$\lambda_{KQ}^{G} = $'+str(list[2])+'\n'+r'$\lambda_{SQ}^{G} = $'+str(list[3]), horizontalalignment='center', verticalalignment='center', fontsize='large')
        plt.text(.82,.82,r'$\kappa_{SK}^{L} = $'+str(list[4])+'\n'r'$\sigma_{KS}^{L} = $'+str(list[5])+'\n'+r'$\lambda_{KQ}^{L} = $'+str(list[6])+'\n'+r'$\lambda_{SQ}^{L} = $'+str(list[7]), horizontalalignment='center', verticalalignment='center', fontsize='large')
        plt.text(.18, .5, r'$\sigma_{SK}^{G} = $' + str(list[8]) + '\n'r'$\kappa_{KS}^{G} = $' + str(list[9]) + '\n' + r'$\kappa_{KQ}^{G} = $' + str(list[10]) + '\n' + r'$\sigma_{SQ}^{G} = $' + str(list[11]),horizontalalignment='center', verticalalignment='center', fontsize='large')
        plt.text(.82, .5, r'$\sigma_{SK}^{L} = $' + str(list[12]) + '\n'r'$\kappa_{KS}^{L} = $' + str(list[13]) + '\n' + r'$\kappa_{KQ}^{L} = $' + str(list[14]) + '\n' + r'$\sigma_{SQ}^{L} = $' + str(list[15]),horizontalalignment='center', verticalalignment='center', fontsize='large')
        plt.text(.18, .25, r'$\kappa^G = $' + str(list[16]) + '\n'r'$\rho^G = $' + str(list[3]+list[11]), horizontalalignment='center', verticalalignment='center', fontsize='large')
        plt.text(.82, .25, r'$\kappa^L = $' + str(list[17]) + '\n'r'$\rho^L = $' + str(list[7] + list[15]), horizontalalignment='center', verticalalignment='center', fontsize='large')
        plt.text(0.5, 0.1, r'$PGR=$' + str(list[18]) + '\n' + r'$PLR=$' + str(list[19]), horizontalalignment='center', verticalalignment='center', fontsize='x-large')
    plt.text(0.5,.01,'Model parameters : '+r'$\beta=$'+str(beta)+(theory=='PT' or theory=='RU')*(', '+r'$\lambda=$'+str(lam))+(theory=='RU')*(', '+r'$\delta=$'+str(delta))+(theory=='EE')*(', '+r'$\eta=$'+str(eta))+(theory=='BRN')*(', '+r'$\alpha=$'+str(alpha)+', '+r'$\alpha^\prime=$'+str(alphaP))+'\nStochastic environment : '+r'$\tau=$'+str(tau)+', '+r'$n=$'+str(n)+' '+r'$p_h=$'+str(ph)+', '+r'$p_l=$'+str(pl)+', '+r'$u=$'+str(u)+', '+r'$d=$'+str(d) + ('\n'+r'$\theta=$'+str(list[-2][0]) + (theory=='LEPT')*(', '+r'$r=$'+str(list[-1])))*(type(list)!=str), horizontalalignment='center', verticalalignment='top')
    #plt.savefig('name.pdf')
    plt.show()
    plt.close()
    return None


#Lagged-expectations PT

def f(beta, lam, w, r):
    theta = thetify(QAOtau(beta, lam, w, r), ph, pl)
    ThetaP_vector = thetify(QAOtauP(beta, lam, w, w * (W_vector - 1), r), ph, pl)
    K, K, ThetaPh, ThetaPl = np.meshgrid(tau_list, tau_list, ThetaP_vector, ThetaP_vector)
    if theta > 0:
        A = Del >= theta  ##investment in the good asset A
        B = Del <= -theta  ##investment in the bad asset B
        pA, pB = (A * P).sum(), (B * P).sum()
        result = (1 - pA - pB)  ##p(O)  ##PB what if we invest at tau' tho...
    else:
        A = Del > 0
        B = Del < 0
        Lottery = 1 - A - B
        result = (P * Wlottery * Lottery * (DelP == 0)).sum() * E_lottery  ##PB theta' can be >0 tho...
        result += (P * Lottery * (.5 * Wh * ((DelP > ThetaPh) + (DelP == ThetaPh) * (ThetaPh > 0)) + .5 * Wl * ((DelP > ThetaPl) + (DelP == ThetaPl) * (ThetaPl > 0)))).sum() * E_Wh
        result += (P * Lottery * (.5 * Wh * ((DelP < -ThetaPh) + (DelP == -ThetaPh) * (ThetaPh > 0)) + .5 * Wl * ((DelP < -ThetaPl) + (DelP == -ThetaPl) * (ThetaPl > 0)))).sum() * E_Wl
    KA = DelP >= ThetaPh  ##the investor keeps conditinal on having invested in A
    SA = (-DelP >= ThetaPh) * (ThetaPh > 0) + (-DelP > 0) * (ThetaPh == 0)
    QA = abs(DelP) < ThetaPh
    KB = -DelP >= ThetaPl
    SB = (DelP >= ThetaPl) * (ThetaPl > 0) + (DelP > 0) * (ThetaPl == 0)
    QB = abs(DelP) < ThetaPl
    result += (P * Wh * A * KA).sum() * E_Wh
    #E[w_T|KA]*p(KA) = E[w'|KA]*p(KA) * E[w'|A] : 2 independent iterations of n price changes with the good asset, while the first n led to reinvest
    result += (P * Wh * A * SA).sum() * E_Wl
    result += (P * Wh * A * QA).sum()
    result += (P * Wl * B * KB).sum() * E_Wl
    result += (P * Wl * B * SB).sum() * E_Wh
    result += (P * Wl * B * QB).sum()
    result = result * w
    #decomposition_validity_test = np.logical_xor(np.logical_xor(np.logical_xor(np.logical_xor(np.logical_xor(KA,SA),QA),KB),SB),QB)==A+B
    return result#, theta, ThetaP_vector[4]


def draw_f(beta, lam, w, r_min, r_max):
    size = 1000
    X = np.linspace(r_min, r_max, size)
    Y = np.array([f(beta, lam, w, r) for r in X])
    # step = (r_max-r_min)/size
    # Z = np.array([(f(beta,lam,w,r+step)-f(beta,lam,w,r-step))/2/step for r in X[1:-1]])  #f'
    plt.figure()
    ax = plt.subplot()
    ax.set_ylim(min(w,w*E_Wl**2)-.1, w*E_Wh**2 + .1)
    ax.plot(X, Y, linestyle='none', marker='.', markersize=1, label=r'$E_0[w_T]$')#, r'$\theta$', 'theta prime'))
    Ones = np.ones(len(X))
    ax.plot(X,X)
    ax.plot(X, w * E_Wh ** 2 * Ones, label=r'$w E[W_h]^2$', linestyle='none', marker='.', markersize=1)
    ax.plot(X, w * E_Wl ** 2 * Ones, label=r'$w E[W_l]^2$', linestyle='none', marker='.', markersize=1)
    ax.plot(X, w * E_lottery ** 2 * Ones, label=r'$w E[W_{lottery}]^2$', linestyle='none', marker='.', markersize=1)
    ax.plot(X, E_optimal_play * Ones, label=r'optimal $E[w_T]$', linestyle='none', marker='.', markersize=1)
    ax.plot(X, w * Ones, label=r'$w$', linestyle='none', marker='.', markersize=1)
    ax.set_xlabel(r'$r_{prior}$')
    ax.set_ylabel(r'$r_{posterior}$')
    ax.set_title('Expected wealth given a fixed reference point')
    ax.legend(numpoints=30)
    # mask = (np.abs(Y[1:-1]-X[1:-1])<(1-Z)*(step))*(np.abs(Z)<1)
    # Attractors = np.trim_zeros(X[1:-1]*mask)
    # Attractors = [r for r in Attractors if r!=0]
    # ax.text(r_max+.1*(r_max-r_min),.5*(r_max+r_min)/2,'Attractors : '+str(Attractors),rotation=-90,horizontalalignment='left', verticalalignment='center', fontsize='large')
    plt.show()
    plt.close()
    # return Attractors

def ref_point(beta, lam, w) :
    start = 3
    while start > 1 :
        r = start*w
        r_next = f(beta,lam,w,r)
        counter = 0
        while True :
            r = r_next
            r_next = f(beta,lam,w,r)
            if counter > 100 :
                break
            if abs(r-r_next)<1e-14 and abs(r-w)>1e-14 :
                return r
            counter += 1
        start -= .1
    if abs(f(beta,lam,w,w)-w)<1e-14 :
        return w




draw_f(beta, lam, w, w, 3)
print(ref_point(beta,lam,w))
propensities_disp(beta,lam,delta,alpha,alphaP,eta,w, 'LEPT')