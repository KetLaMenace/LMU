##quick script to compute sums like benchmark npgr and so on
from math import *
import numpy as np
import matplotlib.pyplot as plt
import time

##binomial coefficient function
def binom(n,k) :
    if k > n or n < 0:
        return 1
    return int(factorial(n)/factorial(k)/factorial(n-k))


##probability that a=k, b=l, a'=m and b'=j under a parametrization of stochatsic environment + agent preferences
def p(tau,n,pa,pb,k,l,m,j) :
    return (binom(tau,k)*binom(tau,l)*binom(n,m-k)*binom(n,j-l))*(pa**m*(1-pa)**(tau+n-m)*pb**j*(1-pb)**(tau+n-j))

##tensor of probabilities done the old but safe way
def Ptest(tau,n,pa,pb) :
    result = np.zeros((tau + 1, tau + 1, n + 1, n + 1))
    for k in range(tau + 1):
        for l in range(tau + 1):
            for kk in range(n + 1):
                for ll in range(n + 1):
                    result[k, l, kk, ll] = p(tau, n, 0.55, 0.45, k, l, k + kk, l + ll)
    return result

##tensor of probabilities with numpy and brains, still partly unexplained
def P(tau,n,pa,pb) :  ##CAUTION indices kk and ll are ups between tau and tau', they correspond to m-k and j-l
    k = np.arange(tau+1)
    kk = np.arange(n+1)
    L, K, KK, LL = np.meshgrid(k, k, kk, kk)   ##UNEXPLAINED : when meshing into a dimension>2, the first 2 indices appear to be switched
    result = np.ones((tau+1,tau+1,n+1,n+1))
    for i in k[1:] :
        result[i,:,:,:] = (result[i-1,:,:,:]*(tau-i+1))/i
        result[:,i,:,:] = (result[:,i-1,:,:]*(tau-i+1))/i
    for i in kk[1:] :
        result[:,:,i,:] = (result[:,:,i-1,:]*(n-i+1))/i
        result[:,:,:,i] = (result[:,:,:,i-1]*(n-i+1))/i
    result = result * (pa ** (K + KK) * (1 - pa) ** (tau + n - K - KK) * pb ** (L + LL) * (1 - pb) ** (tau + n - L - LL))
    return result

##calculation steps to get to the benchmark disposition measure
def NPGR (tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                for j in range(m-theta+1,l+n+1) :
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum

def DPGR (tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                for j in range(l,l+n+1) :
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum

def PGR(tau,n,g,ph,pl,theta) :
    return NPGR(tau,n,g,ph,pl,theta)/DPGR(tau,n,g,ph,pl,theta)

def NPLR (tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                for j in range(m-theta+1,l+n+1) :
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum

def DPLR (tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                for j in range(l,l+n+1) :
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum

def PLR(tau,n,g,ph,pl,theta) :
    return NPLR(tau,n,g,ph,pl,theta)/DPLR(tau,n,g,ph,pl,theta)

##benchmark disposition measures under a parametrization of stochatsic environment + agent preferences
def DM(tau,n,g,ph,pl,theta) :
    return PGR(tau,n,g,ph,pl,theta)-PLR(tau,n,g,ph,pl,theta) , PGR(tau,n,g,ph,pl,theta)/PLR(tau,n,g,ph,pl,theta)-1


##Stochastic possibility calculations
##We compute the possibilities in order of descending delta' values (ascending j values), from V_SQ to V_KQ
def PGV_SQ(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                for j in range(l,min(m-theta+1,l+n+1)) :  ##the benchmark investor keeps for delta' := m-j >= theta
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPGR(tau,n,g,ph,pl,theta)   ##dpgr is simply the probability of gains conditional on an investment in time t=tau. It is also the right denominator here

def PGV_SK(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                for j in range(m-theta+1,m+1) :  ##it is a first order violation to switch for delta' >= 0
                    if j in range(l,l+n+1) : sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPGR(tau,n,g,ph,pl,theta)

def PGV_KS(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                for j in range(m+1,m+theta) :  ##the benchmark investor liquidates for delta' > -theta
                    if j in range(l,l+n+1) : sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPGR(tau,n,g,ph,pl,theta)

def PGV_KQ(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                for j in range(max(m+theta,l),l+n+1) :  ##the benchmark investor switches for delta' <= -theta
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPGR(tau,n,g,ph,pl,theta)


def PLV_SQ(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                for j in range(l,min(m-theta+1,l+n+1)) :
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPLR(tau,n,g,ph,pl,theta)

def PLV_SK(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                for j in range(m-theta+1,m+1) :
                    if j in range(l,l+n+1) : sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPLR(tau,n,g,ph,pl,theta)

def PLV_KS(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                for j in range(m+1,m+theta) :
                    if j in range(l,l+n+1) : sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPLR(tau,n,g,ph,pl,theta)

def PLV_KQ(tau,n,g,ph,pl,theta) :
    sum = 0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                for j in range(max(m+theta,l),l+n+1) :
                    sum += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    return sum/DPLR(tau,n,g,ph,pl,theta)


##returns the vector (PGV_KQ,PGV_KS,PGV_SK,PGV_SQ). The order is now in ascending delta' to match the paper
def PGV(tau,n,g,ph,pl,theta) :
    return PGV_KQ(tau,n,g,ph,pl,theta),PGV_KS(tau,n,g,ph,pl,theta),PGV_SK(tau,n,g,ph,pl,theta),PGV_SQ(tau,n,g,ph,pl,theta)

def PLV(tau,n,g,ph,pl,theta) :
    return PLV_KQ(tau,n,g,ph,pl,theta),PLV_KS(tau,n,g,ph,pl,theta),PLV_SK(tau,n,g,ph,pl,theta),PLV_SQ(tau,n,g,ph,pl,theta)


##some tests
def check(tau,n,g,ph,pl,theta) :
    return PGR(tau,n,g,ph,pl,theta)+PGV(tau,n,g,ph,pl,theta)[3]-1 , PLR(tau,n,g,ph,pl,theta)+PLV(tau,n,g,ph,pl,theta)[3]-1 , PGV(tau,n,g,ph,pl,theta)[0]+PGV(tau,n,g,ph,pl,theta)[1]+PGV(tau,n,g,ph,pl,theta)[2]-PGR(tau,n,g,ph,pl,theta) , PLV(tau,n,g,ph,pl,theta)[0]+PLV(tau,n,g,ph,pl,theta)[1]+PLV(tau,n,g,ph,pl,theta)[2]-PLR(tau,n,g,ph,pl,theta)



##Violation propensity calculations


##relative wealth variation after t time increments with i price increases
def wealth(t,u,d,i) :
    return u**i*d**(t-i)

def g_get(n,u,d) :
    g = 0
    while wealth(n,u,d,g)<1 :
        g+=1
    return g

##getting theta from Q^AO aka "thetifying" the q
def thetify(q,pa,pb) :
    if q<=0 : return int(-1e16)  ##send theta to +- infinity for values of q outside ]0,1[
    if q>=1 : return int(1e16)
    return ceil(log(q/(1-q),pa*(1-pb)/pb/(1-pa)))

##expected value of a random variable taking value L[i] with probability binom(n,i)p^i(1-p)^(n-i)
def E(L,p) :
    sum = 0
    for i in range(len(L)) :
        sum += L[i]*binom(len(L)-1,i)*p**i*(1-p)**(len(L)-1-i)
    return sum

##3.1.1 PT with status-quo reference point (SQPT)
##utility is (x-w)^beta for x>=w and -lam(w-x)^beta otherwise

def v(y,beta,lam) :
    if y>=0 : return y**beta
    return -lam*(-y)**beta

##Q^AO_tau, the analytical formula is -E[v(e_l)]/(E[v(e_h)-E[v(e_l)]) with e_h and e_l the random variables representing gains from the high and low process respectively
def QAOtau_SQPT(beta,lam,n,ph,pl,u,d) :
    return -E([v(wealth(n,u,d,i)-1,beta,lam) for i in range(n+1)] , pl) / ( E([v(wealth(n,u,d,i)-1,beta,lam) for i in range(n+1)] , ph) - E([v(wealth(n,u,d,i)-1,beta,lam) for i in range(n+1)] , pl) )

##Q^AO_tau' with k price increases at time tau and m at time tau'
def QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c) :
    odds = ( v(c,beta,lam) - E([v((c+w)*wealth(n,u,d,i)-w,beta,lam) for i in range(n+1)],pl) ) / ( E([v((c+w)*wealth(n,u,d,i)-w,beta,lam) for i in range(n+1)],ph) - v(c,beta,lam) )
    return odds/(1+odds)

##taylor expansion of Q^AO_tau  !!! NOT VALID !!! (in theory and wrong w)
#def QAOtauP_taylor(beta,lam,tau,n,ph,pl,u,d,k,m) :
#    w = wealth(tau,u,d,k)
#    c = wealth(tau+n,u,d,m) - w
#    odds = - c*E([wealth(n,u,d,k)-1 for k in range(n+1)] , pl)/(c*E([wealth(n,u,d,k)-1 for k in range(n+1)] , ph))
#    return odds/(1+odds)


def kSKG_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)  ##we thetify under the assumption that A is the good asset, without loss of generality (the rest of the function is symmetric A<->B)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)  ##theta' := thetified QAOtau'
                for j in range(m-theta+1,m+1) :
                    if j in range(l,l+n+1) :  ##with these specifications we are in V_SK and G
                        if m-j>=thetaP :
                            numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
                            print(k,l,m,j)  ##the investor invests (keeps) when m-j>=theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

def sKSG_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(m+1,m+theta) :
                    if j in range(l,l+n+1) :  ##with these specifications we are in V_KS and G
                        if j-m>=thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor invests (switches) when j-m>=theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

def lSQG_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(l,min(m-theta+1,l+n+1)) :  ##with these specifications we are in V_SQ and G
                    if m-j<thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor liquidates when |m-j|<theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

##NOT RELEVANT TO PGR
def lKQG_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k+g,k+n+1) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(max(m+theta,l),l+n+1) :  ##with these specifications we are in V_KQ and G
                    if j-m<thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor liquidates when |m-j|<theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

def kSKL_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(m-theta+1,m+1) :
                    if j in range(l,l+n+1) :  ##with these specifications we are in V_SK and L
                        if m-j>=thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor invests (keeps) when m-j>=theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

def sKSL_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(m+1,m+theta) :
                    if j in range(l,l+n+1) :  ##with these specifications we are in V_KS and L
                        if j-m>=thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor invests (switches) when j-m>=theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

def lSQL_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(l,min(m-theta+1,l+n+1)) :  ##with these specifications we are in V_SQ and L
                    if m-j<thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor liquidates when |m-j|<theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator

##NOT RELEVANT TO PLR
def lKQL_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    g = g_get(n,u,d)
    theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
    if theta<=0 : return 0
    numerator,denominator = 0,0
    for k in range(theta,tau+1) :
        for l in range(0,k-theta+1) :
            for m in range(k,k+g) :
                w = wealth(tau,u,d,k)
                c = wealth(tau+n,u,d,m) - w
                thetaP = thetify(QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,w,c),ph,pl)
                for j in range(max(m+theta,l),l+n+1) :  ##with these specifications we are in V_KQ and L
                    if j-m<thetaP : numerator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)  ##the investor liquidates when |m-j|<theta'
                    denominator += p(tau,n,ph,pl,k,l,m,j)+p(tau,n,pl,ph,k,l,m,j)
    if denominator == 0 : return 'not defined'
    return numerator/denominator


def propensities_SQPT(beta,lam,tau,n,ph,pl,u,d) :
    return [kSKG_SQPT(beta,lam,tau,n,ph,pl,u,d),sKSG_SQPT(beta,lam,tau,n,ph,pl,u,d),lSQG_SQPT(beta,lam,tau,n,ph,pl,u,d),lKQG_SQPT(beta,lam,tau,n,ph,pl,u,d),kSKL_SQPT(beta,lam,tau,n,ph,pl,u,d),sKSL_SQPT(beta,lam,tau,n,ph,pl,u,d),lSQL_SQPT(beta,lam,tau,n,ph,pl,u,d),lKQL_SQPT(beta,lam,tau,n,ph,pl,u,d)]

##clean display of all second-order propensities
def propensities_SQPT_disp(beta,lam,tau,n,ph,pl,u,d) :
    list = propensities_SQPT(beta,lam,tau,n,ph,pl,u,d)
    plt.figure()
    plt.axis('off')
    plt.title('Status-quo prospect theory \nSecond-order violation propensities', fontsize='xx-large')
    plt.text(0.25,0.5,r'$\kappa_{SK}^{G} = $'+str(list[0])+'\n'r'$\sigma_{KS}^{G} = $'+str(list[1])+'\n'+r'$\lambda_{SQ}^{G} = $'+str(list[2])+'\n'+r'$\lambda_{KQ}^{G} = $'+str(list[3]), horizontalalignment='center', verticalalignment='center', fontsize='x-large')
    plt.text(0.75,0.5,r'$\kappa_{SK}^{L} = $'+str(list[4])+'\n'r'$\sigma_{KS}^{L} = $'+str(list[5])+'\n'+r'$\lambda_{SQ}^{L} = $'+str(list[6])+'\n'+r'$\lambda_{KQ}^{L} = $'+str(list[7]), horizontalalignment='center', verticalalignment='center', fontsize='x-large')
    plt.text(0.5,0,'Model parameters : '+r'$\beta=$'+str(beta)+', '+r'$\lambda=$'+str(lam)+
    '\nStochastic environment : '+r'$\tau=$'+str(tau)+', '+r'$n=$'+str(n)+' '+r'$p_h=$'+str(ph)+', '+r'$p_l=$'+str(pl)+', '+r'$u=$'+str(u)+', '+r'$d=$'+str(d), horizontalalignment='center', verticalalignment='top')
    plt.show()
    plt.close()
    return None




##lil tool to get the variations of my QAOtau'
def variations(beta,lam,n,ph,pl,u,d,*args) : ##add 2 empty lists to keep the values of c/w and Q
    c = -1+1e-4 ##looking at the range from -1 to very far, and we use w=1 so that c/w can be written c
    Q = QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,1,c)
    Q_right = QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,1,c+1e-4)
    sign_current = copysign(1,Q)
    slope_current = copysign(1,Q_right-Q)
    signs,slopes = [c,sign_current],[c,Q,slope_current]
    if len(args)==2 : c_history,Q_history = args[0],args[1]
    ##signs=[leftmost point of domain, sign of domain, ...] slopes=[leftmost point, value at leftmost point, slope sign, ...] for all domains of constant sign/slope sign
    while(c<1e3) :
        dc = 1e-3 + 1e-3*abs(c)  ##discretion step proportional to distance from 0, with value 1e-4 at 0 and 1e-3 at +-1
        if c>1 : dc = dc*c  ##quadratic with distance for large c
        Q = QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,1,c)
        Q_right = QAOtauP_SQPT(beta,lam,n,ph,pl,u,d,1,c+dc)
        if len(args)==2 :
            if c>1 : c_history.append(log(log(c)+1)+1)
            else : c_history.append(c)
            Q_history.append(Q)
        if copysign(1,Q) != sign_current : ##if Q changed sign
            sign_current = -1*sign_current
            signs.append(c)
            signs.append(sign_current)
        if copysign(1,Q_right-Q) != slope_current and abs(Q_right-Q)>1e-15 : ##if slope changed sign with variation of more than a few bits
            slope_current = -1*slope_current
            slopes.append(c)
            slopes.append(Q)
            slopes.append(slope_current)
        c += dc
    return signs,slopes

def draw(beta,lam_list,n,ph,pl,u,d) :
    N = len(lam_list)
    cmap = plt.get_cmap('inferno_r')
    cindex = 0  ##color index going from 0 to 1 along the lambda list
    color = cmap(.1+.9*cindex)
    X,Y = [],[[]]  ##c/w on the x axis, Q on the y, one Y list per value of lambda
    variations(beta,lam_list[0],n,ph,pl,u,d,X,Y[0])
    ref = QAOtau_SQPT(beta,lam_list[0],n,ph,pl,u,d)
    Y_ref = [ref for c in X]
    plt.figure()
    graph = plt.subplot(111)
    graph.plot(X,Y[0],c=color, label=r'$\lambda =$'+str(lam_list[0]))
    graph.plot(X,Y_ref,c=color,lw=2.5)
    for lam in lam_list[1:] :
        cindex += 1/N
        color = cmap(.1+.9*cindex)
        Y.append([])
        X=[]
        variations(beta,lam,n,ph,pl,u,d,X,Y[-1])
        ref = QAOtau_SQPT(beta,lam,n,ph,pl,u,d)
        Y_ref = [ref for c in X]
        if ceil(2*lam)-2*lam<1e-14 : graph.plot(X,Y[-1],c=color, label=r'$\lambda =$'+str(int(2*lam)/2))  ##Q^AO_tau' plot  //  we add legend when lambda is a semi-integer, shaving off the rounding errors
        else :graph.plot(X,Y[-1],c=color)
        graph.plot(X,Y_ref,c=color,lw=2.5)  ##Q^AO_tau plot
    graph.set_title(r'$Q^{AO}_{\tau\prime}$'+' as a function of c/w for various '+r'$\lambda$')
    graph.set_xlabel('c/w')
    graph.legend()
    plt.show()
    plt.close()
    return None


tau = 2
n = 4
ph = .55
pl = .45
u = 1.3
d = 0.75

w = 1.1
beta = .5
lam = 1


##instruction for plotting QAOtau'(c) with lambda from 1 to 10, or with lambda=2.5
draw(beta,[1/5*i for i in range(5,10*5)],n,ph,pl,u,d)
#draw(1,[2.5],4,0.55,0.45,1.3,0.8)


#propensities_SQPT_disp(beta,lam,tau,n,ph,pl,u,d)

theta = thetify(QAOtau_SQPT(beta,lam,n,ph,pl,u,d),ph,pl)
print(theta)

