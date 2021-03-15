import numpy as np
import pandas as pd
import pickle
import time
from scipy.stats import lognorm
from scipy.stats import gamma
from os import path
from SEIR_viral_load_symptoms import *


###
# ONE SHOT SCREENING
###

def impact_oneshot(inf,dur,onsvals,prob,asymptomatic=0.65,cutoff=6,n_samples=1000,use_pcr=False):
    impacts = np.zeros(len(onsvals))
    for oo, ons in enumerate(onsvals):
        caught = 0
        missed = 0
        for i in range(n_samples):
            c,m = screen_oneshot(inf,dur,ons,prob,asymptomatic,cutoff,use_pcr=use_pcr)
            caught+=c
            missed+=m
        impacts[oo] = caught/(caught+missed)
    return impacts

def screen_oneshot(inf,dur,ons,prob,asymptomatic=0.65,cutoff=6,use_pcr=False):
    if np.random.random()<asymptomatic:
        return screen_oneshot_asymptomatic(inf,dur,ons,prob,cutoff=cutoff,use_pcr=use_pcr)
    else:
        return screen_oneshot_symptomatic(inf,dur,ons,prob,cutoff=cutoff,use_pcr=use_pcr)

def screen_oneshot_symptomatic(inf,dur,ons,prob,cutoff=6,use_pcr=False):
    V,t_symptoms = get_trajectory(is_symptomatic=True)
    I = [inf(x,cutoff=cutoff) for x in V]
    t_test = np.random.choice(t_symptoms)
    if (use_pcr==False):
        if np.random.rand() < prob:
            t_pcr = np.where(V>3)[0][0]
            t_ons = t_pcr+ons
            if (t_test >= t_ons) and (t_test < (t_ons+dur)):
                caught = I[t_test]
                missed = 0
            else:
                caught = 0
                missed = I[t_test]
        else:
            caught = 0
            missed = I[t_test]
    else:
        if use_pcr=='pcr':
            LOD=3
        else:
            LOD=6
        if V[t_test]>LOD:
            caught = I[t_test]
            missed = 0
        else:
            caught = 0
            missed = I[t_test]
    return caught,missed

def screen_oneshot_asymptomatic(inf,dur,ons,prob,cutoff=6,use_pcr=False):
    V = get_trajectory(is_symptomatic=False)
    I = [inf(x,cutoff=cutoff) for x in V]
    t_test = np.random.choice(21)
    if use_pcr==False:
        if np.random.rand() < prob:
            t_pcr = np.where(V>3)[0][0]
            t_ons = t_pcr+ons
            if (t_test >= t_ons) and (t_test < (t_ons+dur)):
                caught = I[t_test]
                missed = 0
            else:
                caught = 0
                missed = I[t_test]
        else:
            caught = 0
            missed = I[t_test]
    else:
        if use_pcr=='pcr':
            LOD=3
        else:
            LOD=6
        if V[t_test]>LOD:
            if np.random.random() < 0.95:
                caught = I[t_test]
                missed = 0
        else:
            caught = 0
            missed = I[t_test]
    return caught,missed

###
# TWO SHOT SCREENING
###

def impact_twoshot(inf,dur,onsvals,prob,asymptomatic=0.65,fpr_smell=0.04,
                   fpr_rapid=0.025,L=6,cutoff=6,n_samples=1000):
    impacts = np.zeros(len(onsvals))
    olfactories = np.zeros(len(onsvals))
    rapids = np.zeros(len(onsvals))
    for oo, ons in enumerate(onsvals):
        caught = 0
        missed = 0
        olfactory = 0
        rapid = 0
        for i in range(n_samples):
            c,m,olf,rap = screen_twoshot(inf,dur,ons,prob,asymptomatic=asymptomatic, 
                                         fpr_smell=fpr_smell,fpr_rapid=fpr_rapid,L=L,cutoff=cutoff)
            caught+=c
            missed+=m
            olfactory += olf
            rapid += rap
        impacts[oo] = caught/(caught+missed)
        olfactories[oo] = olfactory
        rapids[oo] = rapid
    return impacts,olfactories,rapids

def screen_twoshot(inf,dur,ons,prob,asymptomatic=0.65,fpr_smell=0.04,fpr_rapid=0.025,L=6,cutoff=6):
    if np.random.random()<asymptomatic:
        return screen_twoshot_asymptomatic(inf,dur,ons,prob,
                                           fpr_smell=fpr_smell,fpr_rapid=fpr_rapid,
                                           L=L,cutoff=cutoff)
    else:
        return screen_twoshot_symptomatic(inf,dur,ons,prob,
                                           fpr_smell=fpr_smell,fpr_rapid=fpr_rapid,
                                           L=L,cutoff=cutoff)

def screen_twoshot_symptomatic(inf,dur,ons,prob,fpr_smell=0.04,fpr_rapid=0.025,L=6,cutoff=6):
    caught, missed = 0,0
    V,t_symptoms = get_trajectory(is_symptomatic=True)
    I = [inf(x,cutoff=cutoff) for x in V]
    t_test = np.random.choice(t_symptoms)
    # ANOSMIA POSITIVE PERSON
    if np.random.rand() < prob:
        t_pcr = np.where(V>3)[0][0]
        t_ons = t_pcr+ons
        # ACUTE ANOSMIA
        if (t_test >= t_ons) and (t_test < (t_ons+dur)):
            fails_olfaction = True
        # NO ACUTE ANOSMIA
        else:
            # FALSE POSITIVE USMELLIT
            if np.random.rand()<fpr_smell:
                fails_olfaction = True
            # TRUE NEGATIVE USMELLIT
            else:
                fails_olfaction = False
    # ANOSMIA NEGATIVE PERSON
    else:
        # FALSE POSITIVE USMELLIT
        if np.random.rand()<fpr_smell:
            fails_olfaction = True
        # TRUE NEGATIVE USMELLIT
        else:
            fails_olfaction = False
    
    fails_rapid = False
    if fails_olfaction == True:
        if (V[t_test]>=L) or (np.random.rand() < fpr_rapid):
            fails_rapid = True
    
    if fails_olfaction and fails_rapid:
        caught = I[t_test]
    else:
        missed = I[t_test]
        
    return caught,missed,fails_olfaction,fails_rapid

def screen_twoshot_asymptomatic(inf,dur,ons,prob,fpr_smell=0.04,fpr_rapid=0.025,L=6,cutoff=6):
    caught, missed = 0,0
    V = get_trajectory(is_symptomatic=False)
    I = [inf(x,cutoff=cutoff) for x in V]
    t_test = np.random.choice(21)
    # ANOSMIA POSITIVE PERSON
    if np.random.rand() < prob:
        t_pcr = np.where(V>3)[0][0]
        t_ons = t_pcr+ons
        # ACUTE ANOSMIA
        if (t_test >= t_ons) and (t_test < (t_ons+dur)):
            fails_olfaction = True
        # NO ACUTE ANOSMIA
        else:
            # FALSE POSITIVE USMELLIT
            if np.random.rand()<fpr_smell:
                fails_olfaction = True
            # TRUE NEGATIVE USMELLIT
            else:
                fails_olfaction = False
    # ANOSMIA NEGATIVE PERSON
    else:
        # FALSE POSITIVE USMELLIT
        if np.random.rand()<fpr_smell:
            fails_olfaction = True
        # TRUE NEGATIVE USMELLIT
        else:
            fails_olfaction = False
            
    fails_rapid = False
    if fails_olfaction == True:
        if (V[t_test]>=L) or (np.random.rand() < fpr_rapid):
            fails_rapid = True
    
    if fails_olfaction and fails_rapid:
        caught = I[t_test]
    else:
        missed = I[t_test]
        
    return caught,missed,fails_olfaction,fails_rapid

###
# REPEATED SCREENING
###

def infectiousness_removed_indiv(inf,dur,ons,prob,D,asymptomatic=0.65,cutoff=6,use_pcr=False):
    if np.random.random()<asymptomatic:
        a,b,c = infectiousness_removed_indiv_asymptomatic(inf,dur,ons,prob,D,cutoff=cutoff,use_pcr=use_pcr)
    else:
        a,b,c = infectiousness_removed_indiv_symptomatic(inf,dur,ons,prob,D,cutoff=cutoff,use_pcr=use_pcr)
    return a,b,c

def infectiousness_removed_indiv_symptomatic(inf,dur,ons,prob,D,cutoff=6,use_pcr=False):
    V,t_symptoms = get_trajectory(is_symptomatic=True)
    I = [inf(x,cutoff=cutoff) for x in V]
    total = np.sum(I)
    phase = np.random.choice(D)
    removed_by_testing = 0
    removed_by_symptoms = 0
    
    t_test = np.arange(phase,28,D)
    dt=0
    if use_pcr=='pcr':
        pos_by_pcr = np.where(V>3)[0]
        t_pos = np.intersect1d(t_test,pos_by_pcr)
        dt=1
    elif use_pcr=='rapid_ag':
        pos_by_pcr = np.where(V>6)[0]
        t_pos = np.intersect1d(t_test,pos_by_pcr)
        dt=0
    else:
        t_pcr = np.where(V>3)[0][0]
        if np.random.rand() < prob:
            t_ons = t_pcr+ons
            t_symptomatic = np.arange(t_ons,t_ons+dur)
            t_pos = np.intersect1d(t_test,t_symptomatic)
        else:
            t_pos = []
    
    if len(t_pos)==0:
        removed_by_testing = 0
        removed_by_symptoms = np.sum(I[t_symptoms:])
    else:
        t_dx = t_pos[0]+dt
        if t_dx == t_symptoms:
            # break ties with a coin flip
            if np.random.rand()<0.5:
                removed_by_testing = np.sum(I[t_dx:])
                removed_by_symptoms = 0
            else:
                removed_by_testing = 0
                removed_by_symptoms = np.sum(I[t_dx:])
        elif t_dx < t_symptoms:
            removed_by_testing = np.sum(I[t_dx:])
            removed_by_symptoms = 0
        elif t_dx > t_symptoms:
            removed_by_testing = 0
            removed_by_symptoms = np.sum(I[t_symptoms:])
    return removed_by_testing,removed_by_symptoms,total

def infectiousness_removed_indiv_asymptomatic(inf,dur,ons,prob,D,cutoff=6,use_pcr=False):
    V = get_trajectory(is_symptomatic=False)
    I = [inf(x,cutoff=cutoff) for x in V]
    total = np.sum(I)
    phase = np.random.choice(D)
    removed_by_testing = 0
    removed_by_symptoms = 0

    t_test = np.arange(phase,28,D)
    dt=0
    
    if use_pcr=='pcr':
        pos_by_pcr = np.where(V>3)[0]
        t_pos = np.intersect1d(t_test,pos_by_pcr)
        dt=1
    elif use_pcr=='rapid_ag':
        pos_by_pcr = np.where(V>6)[0]
        t_pos = np.intersect1d(t_test,pos_by_pcr)
        dt=0
    else:
        t_pcr = np.where(V>3)[0][0]
        if np.random.rand() < prob:
            t_ons = t_pcr+ons
            t_symptomatic = np.arange(t_ons,t_ons+dur)
            t_pos = np.intersect1d(t_test,t_symptomatic)
        else:
            t_pos = []

    if len(t_pos)==0:
        removed_by_testing = 0
    else:
        t_dx = t_pos[0]+dt
        removed_by_testing = np.sum(I[t_dx:])
        removed_by_symptoms = 0
    return removed_by_testing,removed_by_symptoms,total

def get_R_reduction_factor(inf,dur,ons,prob,D,asymptomatic=0.65,cutoff=6,n_samples=10000,use_pcr=False):
    pol,no_pol = 0,0
    for i in range(n_samples):
        a,b,c = infectiousness_removed_indiv(inf,dur,ons,prob,D,asymptomatic=asymptomatic,cutoff=cutoff,use_pcr=use_pcr)
        no_pol += (c-b)
        pol += (c-(a+b))
    return pol/no_pol

def impact_population(inf,dur,onsvals,prob,Dvals,asymptomatic,n_samples,use_pcr=False):
    test = np.zeros([len(onsvals),len(Dvals)])
    self = np.zeros([len(onsvals),len(Dvals)])
    R_reduction_factor = np.zeros([len(onsvals),len(Dvals)])
    for oo,ons in enumerate(onsvals):
        for dd,D in enumerate(Dvals):
            _test, _self, _nopol = infectiousness_removed_pop(inf,dur,ons,prob,D,asymptomatic=asymptomatic,n_samples=n_samples,use_pcr=use_pcr)
            test[oo,dd] = _test/_nopol
            self[oo,dd] = _self/_nopol
            R_reduction_factor[oo,dd]=get_R_reduction_factor(inf,dur,ons,prob,D,asymptomatic=asymptomatic,n_samples=n_samples,use_pcr=use_pcr)        
    notest,noself,nototal = infectiousness_removed_pop(inf,1,15,0,100000,asymptomatic=asymptomatic,n_samples=n_samples,use_pcr=use_pcr)
    notest /= nototal
    noself /= nototal
    return R_reduction_factor,test,self,notest,noself

def infectiousness_removed_pop(inf,dur,ons,prob,D,asymptomatic=0.65,cutoff=6,n_samples = 1000,use_pcr=False):
    test,self,total = 0,0,0
    for i in range(n_samples):
        a,b,c = infectiousness_removed_indiv(inf,dur,ons,prob,
                                             D,asymptomatic=asymptomatic,cutoff=cutoff,use_pcr=use_pcr)
        test+=a
        self+=b
        total+=c
    return test/n_samples,self/n_samples,total/n_samples










