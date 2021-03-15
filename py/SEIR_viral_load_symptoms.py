import numpy as np
import pandas as pd
import pickle
import time
from scipy.stats import lognorm
from scipy.stats import gamma
from os import path


def VL(t,T3,TP,YP,T6):
    '''
    VL(t,T3,TP,YP,T6)
        Computes the scalar value of a hinge function with control points:
        (T3, 3)
        (TP, YP)
        (T6, 6)
    Returns zero whenever (1) t<T3 or (2) hinge(t) is negative.
    '''
    if t < T3:
        return 0
    if t < TP:
        return (t-T3)*(YP-3)/(TP-T3)+3
    return np.max([(t-TP)*(6-YP)/(T6-TP)+YP,0])
    
def get_trajectory(is_symptomatic=False,full=False):
    '''
    get_trajectory(is_symptomatic=False,full=False)
        Stochastically draws a viral load trajectory by randomly drawing control points.
    
    is_symptomatic allows separate models for symptomatic and asymptomatic trajectories which
    the review by Cevik et al 2020 has shown to be different. (Asymptomatic clearance is faster)
    
    When full==False, returns a viral load trajectory evaluated on int days 0,1,2,...,27,
    where day 0 is the day of exposure. 

    When full==True, returns the integer-evaluated trajectory, as well as:
    v_fine the same trajectory evaluated at values between 0 and 28, with step size 0.01
    t_fine the steps at which the trajectory was evaluated
    t_3, t_peak, v_peak, t_6, the four stochastically drawn control values

    Use full==True for plotting; use full==False for more rapid simulation.
    '''
    if is_symptomatic==True:
        return get_symptomatic_trajectory(full=full)
    else:
        return get_asymptomatic_trajectory(full=full)

def get_symptomatic_trajectory(full=False):
    # Draw control points. 
    # Truncate so that max of t_peak is 3.
    t_peak = gamma.rvs(2.5,loc=0.5)
    while t_peak > 4:
        t_peak = gamma.rvs(2.5,loc=0.5)
    v_peak = np.random.uniform(7,11)
    t_6 = np.random.uniform(4,8)
    t_3 = np.random.random()+2.5
    t_symptoms = np.random.uniform(0,2)
    # Compute t and v vectors and return 
    t = np.arange(28)
    v = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak+t_symptoms) for x in t])
    if full==True:
        tfine = np.arange(0,28,0.01)
        vfine = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak+t_symptoms) for x in tfine])
        return v,vfine,tfine,t_3,t_peak,v_peak,t_6,t_symptoms
    return v,int(np.round(t_symptoms+t_3+t_peak))

def get_asymptomatic_trajectory(full=False):    
    # Draw control points. 
    # Truncate so that max of t_peak is 3.
    t_peak = gamma.rvs(2.5,loc=0.5)
    while t_peak > 4:
        t_peak = gamma.rvs(2.5,loc=0.5)
    v_peak = np.random.uniform(7,11)
    t_6 = np.random.uniform(4,8)
    t_3 = np.random.random()+2.5
    # Compute t and v vectors and return 
    t = np.arange(28)
    v = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak) for x in t])
    if full==True:
        tfine = np.arange(0,28,0.01)
        vfine = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak) for x in tfine])
        return v,vfine,tfine,t_3,t_peak,v_peak,t_6
    return v

# INFECTIOUSNESS FUNCTIONS

def proportional(v,cutoff=6):
    '''
    proportional(v,cutoff=6)
        returns the infectiousness of a viral load v
        for the proportional model.
    '''
    if v < cutoff:
        return 0
    else:
        x = 10**(v-cutoff)
        return x

def threshold(v,cutoff=6):
    '''
    threshold(v,cutoff=6)
        returns the infectiousness of a viral load v
        for the threshold model.
    '''
    if v < cutoff:
        return 0
    else:
        return 1

def logproportional(v,cutoff=6):
    '''
    logproportional(v,cutoff=6)
        returns the infectiousness of a viral load v
        for the logproportional model
        (Manuscript default)
    '''
    if v < cutoff:
        return 0
    else:
        return v-cutoff

# SEIR FULLY MIXED MODEL

def compute_factor_to_calibrate_R0_SQ(infectiousness,asymptomatic,cutoff):
    '''
    compute_factor_to_calibrate_R0_SQ(infectiousness,asymptomatic,cutoff)
        infectiousness: a function handle {proportional,logproportional,threshold}
        asymptomatic: the fraction of individuals who are asymptomatic [0,1]
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)

    Returns a constant mean_infectiousness which is used to scale absolute infectiousness in simulation
    '''
    total_infectiousness = 0
    n_draws = 10000

    loads_asymptomatic = pickle.load(open('asymptomatic_loads.pkl','rb'))
    loads_symptomatic = pickle.load(open('symptomatic_loads.pkl','rb'))
    len_list_a = len(loads_asymptomatic)
    len_list_s = len(loads_symptomatic)

    for i in range(n_draws):
        if np.random.random() < asymptomatic:
            VL,IN,total = loads_asymptomatic[np.random.randint(len_list_a)]
            total_infectiousness += np.sum(IN)
        else:
            VL,t_symptoms,IN,total = loads_symptomatic[np.random.randint(len_list_s)]
            total_infectiousness += np.sum(IN[:t_symptoms])
    mean_infectiousness = total_infectiousness/n_draws
    return mean_infectiousness

def get_Reff(N,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6):
    '''
    get_Reff(N,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6)
        N: population size
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        infectiousness_function: a function handle {proportional,logproportional,threshold}
        asymptomatic: fraction asymptomatic
        results_delay: fixed time delay to return results
        R0: basic reproductive number
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
    '''
    I_init=int(0.005*N)
    _,_,_,_,_,_,Iint = SEIRsimulation(
        N=N,
        external_rate=0,
        D=D,
        L=L,
        infectiousness_function=infectiousness_function,
        asymptomatic=asymptomatic,
        results_delay=results_delay,
        R0=R0,
        cutoff=cutoff,
        I_init=I_init,
        tmax=15,
        calibration_mode=True
        )
    return Iint/I_init

def SEIRsimulation(N,external_rate,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6,I_init=0,tmax=365,calibration_mode=False):
    loads_asymptomatic = pickle.load(open('asymptomatic_loads.pkl','rb'))
    loads_symptomatic = pickle.load(open('symptomatic_loads.pkl','rb'))
    len_list_a = len(loads_asymptomatic)
    len_list_s = len(loads_symptomatic)

    c = compute_factor_to_calibrate_R0_SQ(infectiousness_function,asymptomatic,cutoff)
    k = R0/(c*(N-1))
    
    was_infected_ext = np.zeros(N)
    was_infected_int = np.zeros(N)
    
    is_symptomatic = np.random.binomial(1,1-asymptomatic,size=N)
    t_symptoms = np.zeros(N)
    viral_loads = {}
    infectious_loads = {}
    for i in range(N):
        if is_symptomatic[i]:
            viral_loads[i],t_symptoms[i],infectious_loads[i],_ = loads_symptomatic[np.random.randint(len_list_s)]
        else:
            viral_loads[i],infectious_loads[i],_ = loads_asymptomatic[np.random.randint(len_list_a)]
        infectious_loads[i] = infectious_loads[i]*k

    test_schedule = np.random.choice(D,size=N)

    S = set(np.arange(N))
    I = set()
    R = set()
    Q = set()
    SQ = set()
    St = []
    It = []
    Rt = []
    Qt = []
    SQt = []

    infection_day = np.zeros(N,dtype=int)
    days_till_results = -1*np.ones(N)

    for t in range(tmax):
        # Test
        tested = np.where((test_schedule + t) % D==0)[0]
        for i in tested:
            if (i in Q) or (i in R) or (i in SQ):
                continue
            if days_till_results[i] > -1:
                continue
            if viral_loads[i][infection_day[i]] > L:
                days_till_results[i] = results_delay

        # Isolate
        for i in I:
            if days_till_results[i]==0:
                I = I - set([i])
                Q = Q.union(set([i]))
        
        # Self Quarantine
        for i in I:
            if is_symptomatic[i]==1:
                if infection_day[i]==t_symptoms[i]:
                    I = I-set([i])
                    SQ = SQ.union(set([i]))

        # Initial infections
        if t==0:
            initial_infections = np.random.choice(list(S),size=I_init,replace=False)
            was_infected_ext[initial_infections] = 1    
            S = S - set(initial_infections)
            I = I.union(set(initial_infections))

        # External infections
        ext_infections = np.random.choice(list(S),np.random.binomial(len(S),external_rate))
        was_infected_ext[ext_infections] = 1
        S = S - set(ext_infections)
        I = I.union(set(ext_infections))

        # Internal infections
        infectiousnesses = [infectious_loads[i][infection_day[i]] for i in I]
        p_int_infection = 1 - np.prod(1-np.array(infectiousnesses))
        # print(len(I),p_int_infection)
        int_infections = np.random.choice(list(S),np.random.binomial(len(S),p_int_infection))
        was_infected_int[int_infections] = 1
        S = S - set(int_infections)
        if calibration_mode==False:
            I = I.union(set(int_infections))
        else:
            Q = Q.union(set(int_infections))

        # Update all infection days
        for i in I:
            if (infection_day[i] == 27) or ((viral_loads[i][infection_day[i]]<6) and (infection_day[i]>7)):
                I = I - set([i])
                R = R.union(set([i]))
            infection_day[i] += 1
        for q in Q:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                Q = Q - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        for q in SQ:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                SQ = SQ - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        
        # Update the results delays:
        for i in range(N):
            if days_till_results[i] > -1:
                days_till_results[i] = days_till_results[i] - 1
        
        St.append(len(S))
        It.append(len(I))
        Rt.append(len(R))
        Qt.append(len(Q))
        SQt.append(len(SQ))
    return St,It,Rt,Qt,SQt,np.sum(was_infected_ext),np.sum(was_infected_int)

def SEIRsimulation_suppression(N,
                               external_rate,
                               dur,ons,prob_anosmia,
                               D,L,dt,
                               infectiousness_function,
                               prev_cutoff,
                               se_sample,
                               refusal_rate,
                               asymptomatic=0.65,
                               R0=2.5,
                               cutoff=6,
                               I_init=0,
                               tmax=120,
                               calibration_mode=False):
    loads_asymptomatic = pickle.load(open('asymptomatic_loads.pkl','rb'))
    loads_symptomatic = pickle.load(open('symptomatic_loads.pkl','rb'))
    len_list_a = len(loads_asymptomatic)
    len_list_s = len(loads_symptomatic)

    if prev_cutoff==0:
        is_control_on = True
    else:
        is_control_on = False

    c = compute_factor_to_calibrate_R0_SQ(infectiousness_function,asymptomatic,cutoff)
    k = R0/(c*(N-1))
    
    was_infected_ext = np.zeros(N)
    was_infected_int = np.zeros(N)
    pre_control_internal = np.sum(was_infected_int)
    pre_control_external = np.sum(was_infected_ext)
    
    is_refuser = np.random.binomial(1,refusal_rate,size=N)
    
    is_symptomatic = np.random.binomial(1,1-asymptomatic,size=N)
    t_symptoms = np.zeros(N)
    
    is_anosmia = np.random.binomial(1,prob_anosmia,size=N)
    
    viral_loads = {}
    infectious_loads = {}
    onsets = np.zeros(N)
    for i in range(N):
        if is_symptomatic[i]:
            viral_loads[i],t_symptoms[i],infectious_loads[i],_ = loads_symptomatic[np.random.randint(len_list_s)]
        else:
            viral_loads[i],infectious_loads[i],_ = loads_asymptomatic[np.random.randint(len_list_a)]
        # Rescale infectious_loads:
        infectious_loads[i] = infectious_loads[i]*k
        onsets[i] = np.where(viral_loads[i]>3)[0][0] + ons

    if D != 0:
        test_schedule = np.random.choice(D,size=N)
        is_oneshot = False
        has_used_oneshot = False
    elif D==0:
        D = 1
        test_schedule = np.random.choice(D,size=N)
        is_oneshot = True
        has_used_oneshot = False

    S = set(np.arange(N))
    I = set()
    R = set()
    Q = set()
    SQ = set()
    St = []
    It = []
    Rt = []
    Qt = []
    SQt = []
    ODt = np.zeros(tmax)
    molt = np.zeros(tmax)

    infection_day = np.zeros(N,dtype=int)
    days_till_results = -1*np.ones(N)
    
    for t in range(tmax):
        
        if (is_control_on==True) and (is_oneshot==True):
            is_control_on = False
            has_used_oneshot = True
            
        if ((len(I)+len(Q)+len(SQ))/N > prev_cutoff) and (is_control_on==False):
            if is_oneshot==True and has_used_oneshot==True:
                is_control_on=False
            else:
                is_control_on=True
                pre_control_internal = np.sum(was_infected_int)
                pre_control_external = np.sum(was_infected_ext)
        
        if is_control_on==True:
            # Test
            tested = np.where(((test_schedule + t) % D)==0)[0]
            for i in tested:
                if is_refuser[i]:
                    continue
                if (i in Q) or (i in R) or (i in SQ):
                    continue
                if days_till_results[i] > -1: # don't get tested while waiting for results
                    continue
                
                ODt[t] += 1
                if is_anosmia[i]:
                    if (infection_day[i]>=onsets[i]) and (infection_day[i]<(onsets[i]+dur)): #positive OD test
                        days_till_results[i] = 0
                        molt[t] += 1
                
                if dur==0 and ons==0: # SLOPPY WAY TO USE PCR TESTING.
                    if viral_loads[i][infection_day[i]] > L:
                        if np.random.random() < se_sample:
                            days_till_results[i] = dt

        # Isolate
        for i in I:
            if days_till_results[i]==0:
                I = I - set([i])
                Q = Q.union(set([i]))
        
        # Self Quarantine
        for i in I:
            if is_symptomatic[i]==1:
                if infection_day[i]==t_symptoms[i]:
                    I = I-set([i])
                    SQ = SQ.union(set([i]))

        # Initial infections
        if t==0:
            initial_infections = np.random.choice(list(S),size=I_init,replace=False)
            was_infected_ext[initial_infections] = 1    
            S = S - set(initial_infections)
            I = I.union(set(initial_infections))

        # External infections
        ext_infections = np.random.choice(list(S),np.random.binomial(len(S),external_rate))
        was_infected_ext[ext_infections] = 1
        S = S - set(ext_infections)
        I = I.union(set(ext_infections))

        # Internal infections
        infectiousnesses = [infectious_loads[i][infection_day[i]] for i in I]
        p_int_infection = 1 - np.prod(1-np.array(infectiousnesses))
        # print(len(I),p_int_infection)
        int_infections = np.random.choice(list(S),np.random.binomial(len(S),p_int_infection))
        was_infected_int[int_infections] = 1
        S = S - set(int_infections)
        if calibration_mode==False:
            I = I.union(set(int_infections))
        else:
            Q = Q.union(set(int_infections))

        # Update all infection days
        for i in I:
            if (infection_day[i] == 27) or ((viral_loads[i][infection_day[i]]<6) and (infection_day[i]>7)):
                I = I - set([i])
                R = R.union(set([i]))
            infection_day[i] += 1
        for q in Q:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                Q = Q - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        for q in SQ:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                SQ = SQ - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        
        # Update the results delays:
        for i in range(N):
            if days_till_results[i] > -1:
                days_till_results[i] = days_till_results[i] - 1
        
        St.append(len(S))
        It.append(len(I))
        Rt.append(len(R))
        Qt.append(len(Q))
        SQt.append(len(SQ))
                
    return St,It,Rt,Qt,SQt,np.sum(was_infected_ext),np.sum(was_infected_int),pre_control_external,pre_control_internal,ODt,molt