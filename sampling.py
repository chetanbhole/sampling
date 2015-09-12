# written by Chetan Bhole @ 2015
# Tutorial code to understand different types of sampling algorithms
# The proposal and target distributions are simple 1d distributions mostly.

from argparse import ArgumentParser
import sys
import traceback
import numpy as np
import math

import matplotlib.pyplot as plt
from pylab import *

import scipy.special as sps
from scipy.stats import cauchy
from scipy.stats import gamma
from scipy.stats import uniform

# so legend plots one point instead of two points for each marker
rcParams['legend.numpoints'] = 1

## Seperate out plot bounds and computation bounds
## since most continuous distributions would extend from -inf to +inf, you do need a cut-off
PLOT_LBOUND = -40
PLOT_RBOUND =  40

COMP_LBOUND = -40
COMP_RBOUND =  40

# all unwanted plotting goes to this figure which is not displayed
DUMMY_FIG = 100

# Note for gamma distribution
# the mean of gamma distribution is
# GAMMA_MEAN + GAMMA_SHAPE*GAMMA_SCALE

class Distribution(object):
    rv = 0
    fn_scale = 0

    def pdf(self, x): 
        return self.rv.pdf(x)

    def fn_pdf(self, x):
        return self.rv.pdf(x)*self.fn_scale

    def rvs(self, num_samples): 
        return self.rv.rvs(size=num_samples)

    # Create based on class name:
    def factory(type, shape, loc, scale, fn_scale):
        #return eval(type + "()")
        if type == "cauchy": 
            return CauchyD(shape, loc, scale, fn_scale)
        if type == "gamma": 
            return GammaD(shape, loc, scale, fn_scale)
        if type == "uniform":
            return UniformD(shape, loc, scale, fn_scale)
        assert 0, "Bad distribution creation: " + type
    factory = staticmethod(factory)

class CauchyD(Distribution):
    def __init__(self, shape, loc, scale, fn_scale):
        self.rv = cauchy(loc=loc, scale=scale)
        self.fn_scale = fn_scale    

class GammaD(Distribution):
    def __init__(self, shape, loc, scale, fn_scale):
        self.rv = gamma(shape, loc=loc, scale=scale)
        self.fn_scale = fn_scale    

class UniformD(Distribution):
    def __init__(self, shape, loc, scale, fn_scale):
        self.rv = uniform(loc=loc, scale=scale)
        self.fn_scale = fn_scale    


class DistributionPlotter:
    def generate_samples(self, distr, num_samples, truncate=True):
        s_points = distr.rvs(num_samples)
            
        if truncate:
            s_points = s_points[(s_points > PLOT_LBOUND) & (s_points < PLOT_RBOUND)] 
            
        return s_points

    def plot_hist(self, fig_no, title, points, num_bins):
        plt.figure(fig_no)
        point_bags = plt.hist(points, num_bins)
        plt.title(title)
        
    def plot_norm_hist(self, fig_no, title, points, num_bins, color='b'):
        ## use np.histogram instead :) to avoid the hist plotting
        fig = plt.figure(DUMMY_FIG) # will plot in some fig we don't care about
        point_bags = plt.hist(points, num_bins)
        plt.close(fig)

        # assuming at least 2 bins exist
        bin_width = point_bags[1][1] - point_bags[1][0]

        total_points = len(points)
        prob_points = point_bags[0]/(total_points*bin_width)

        # actual plotting
        plt.figure(fig_no)
        plot_line, = plt.plot(point_bags[1][:num_bins]+bin_width/2.0, prob_points, lw=3, color=color, label=title)
        return plot_line

    def plot_scaled_pdf(self, distr, fig_no, title, scaled=True, color='b'):
        # truncate always true for plotting pdf
        num_pts = 10000
        plt.figure(fig_no)
        x = np.linspace(PLOT_LBOUND, PLOT_RBOUND, num_pts)
        if scaled:
            plot_line, = plt.plot(x, distr.fn_pdf(x), color=color, lw=4, alpha=0.8, label=title)
        else:
            plot_line, = plt.plot(x, distr.pdf(x), color=color, lw=4, alpha=0.8, label=title)
        return plot_line


class Sampling:
    def __init__(self,
                 target, target_shape, target_loc, target_scale, target_fn_scale,
                 proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale):
        
        # distribution variables
        self.target_distr_str = target
        self.proposal_distr_str = proposal

        # values for parameters of distributions
        self.target_shape = target_shape
        self.target_loc = target_loc
        self.target_scale = target_scale
        self.target_fn_scale = target_fn_scale

        self.proposal_shape = proposal_shape
        self.proposal_loc = proposal_loc
        self.proposal_scale = proposal_scale
        self.proposal_fn_scale = proposal_fn_scale

        self.target = Distribution.factory(target, target_shape, target_loc, target_scale, target_fn_scale)
        self.proposal = Distribution.factory(proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale)

        # for samples, keep number of samples the same
        self.num_samples = 10000
        self.num_samples_int = 200 # for interactive mode
        self.num_bins = 50


    def visualize_basic_plots(self):

        dpltr = DistributionPlotter()
        fig_no = 1
        points_target = dpltr.generate_samples(self.target, self.num_samples)
        dpltr.plot_hist(fig_no, 'histogram - ' + self.target_distr_str, points_target, self.num_bins)
        
        fig_no = 2
        points_proposal = dpltr.generate_samples(self.proposal, self.num_samples)
        dpltr.plot_hist(fig_no, 'histogram - ' + self.proposal_distr_str, points_proposal, self.num_bins)


        fig_no = 3
        target_line = dpltr.plot_norm_hist(fig_no, 'normalized histogram - ' + self.target_distr_str, 
                                           points_target, self.num_bins, color='b')
        proposal_line = dpltr.plot_norm_hist(fig_no, 'normalized histogram - ' + self.proposal_distr_str, 
                                             points_proposal, self.num_bins, color='r')
        plt.figure(fig_no)
        plt.legend([target_line, proposal_line])

        fig_no = 4
        target_line = dpltr.plot_scaled_pdf(self.target, fig_no, self.target_distr_str + ' pdf', scaled=False, color='b')
        proposal_line = dpltr.plot_scaled_pdf(self.proposal, fig_no, self.proposal_distr_str + ' pdf', scaled=False, color='r')
        plt.figure(fig_no)
        plt.legend([target_line, proposal_line])

        fig_no = 5
        target_line = dpltr.plot_scaled_pdf(self.target, fig_no, self.target_distr_str + ' fn scaled pdf', color='b')
        proposal_line = dpltr.plot_scaled_pdf(self.proposal, fig_no, self.proposal_distr_str + ' fn scaled pdf', color='r')
        plt.figure(fig_no)
        plt.legend([target_line, proposal_line])

        plt.show()

    def print_info(self):
        print 'target distribution is ', self.target_distr_str
        print 'proposal distribution is ', self.proposal_distr_str



class Rejection(Sampling):
    def __init__(self,
                 target, target_shape, target_loc, target_scale, target_fn_scale,
                 proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale):
        Sampling.__init__(self,
                          target, target_shape, target_loc, target_scale, target_fn_scale,
                          proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale)    


    def generate_target_samples(self, interactive=False):

        dpltr = DistributionPlotter()
        fig_no = 1
        target_line = dpltr.plot_scaled_pdf(self.target, fig_no, self.target_distr_str + 'fn scaled pdf', color='b')
        proposal_line = dpltr.plot_scaled_pdf(self.proposal, fig_no, self.proposal_distr_str + ' fn scaled pdf', color='r')
        all_legend = [target_line, proposal_line]
        plt.figure(fig_no)
        plt.legend(all_legend)
        
        num_samples = 0
        if interactive:
            num_samples = self.num_samples_int
        else:
            num_samples = self.num_samples

        self.s_proposal = dpltr.generate_samples(self.proposal, num_samples)

        unif_d = Distribution.factory("uniform", 0, 0, 1.0, 1.0)
        s_unif = dpltr.generate_samples(unif_d, len(self.s_proposal))

        ion()
        plt.figure(fig_no)

        s_keep = []
        ptno = 0
        first_pass_sample = True
        first_pass_accept = True
        for sm in self.s_proposal:
            if interactive:
                line1, = plt.plot(sm, 0, c='g', marker='+', markersize=15, label='sample from proposal')
                sample_str = 'sample ' + str(sm)
                plt.title(sample_str)
                if first_pass_sample:
                    all_legend.append(line1)
                    first_pass_sample = False
                    plt.legend(all_legend)
                draw()
                pause(0.001)

            t_pdf = self.target.pdf(sm)
            p_pdf = self.proposal.pdf(sm)

            if (s_unif[ptno]*p_pdf*self.proposal_fn_scale <= t_pdf):
                s_keep.append(sm)
                if interactive:
                    line2, = plt.plot(sm, 0, c='r', marker='*', markersize=15, label='accepted sample')
                    accepted_str = 'accepted ' + str(sm)
                    plt.title(accepted_str)
                    if first_pass_accept:
                        all_legend.append(line2)
                        first_pass_accept = False
                        plt.legend(all_legend)
                    draw()
                    pause(0.001)

            ptno = ptno + 1

        hist_line = dpltr.plot_norm_hist(fig_no, 'sampled norm histogram', s_keep, self.num_bins, color='y')
        all_legend.append(hist_line)
        plt.legend(all_legend)
        plt.show()
        ioff()


class Importance(Sampling):
    def __init__(self,
                 target, target_shape, target_loc, target_scale, target_fn_scale,
                 proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale):
        Sampling.__init__(self,
                          target, target_shape, target_loc, target_scale, target_fn_scale,
                          proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale)    
        
        # brute force value
        self.E_fn = 0
        # sampling value
        self.E_fn_sampling = 0

    # function whose expectation needs to be found
    def function(self, x):
        return -0.05 + x*1e-3 + 0.05*sin(0.2*x+5)

    def function_x(self, x):
        return x

    def function_1(self, x):
        return np.ones(len(x))

    def plot_function(self, fig_no, title='function', color='g', truncate=True):
        num_pts = 10000
        plt.figure(fig_no)
        x = np.linspace(PLOT_LBOUND, PLOT_RBOUND, num_pts)
        pline, = plt.plot(x, self.function(x), color=color, lw=2, alpha=0.8, label=title)
        return pline
        
    # This is a function that bruteforces by taking lots of values along a grid
    # and finds the expectation.
    def compute_E_fn(self):
        # truncate will always be true
        num_pts = 100000
        x = np.linspace(COMP_LBOUND, COMP_RBOUND, num_pts)
        bin_width = (COMP_RBOUND - COMP_LBOUND)/(1.0*num_pts) 
        fn_vals = self.function(x)
        pdf_vals = self.target.pdf(x)
        # we convert the pdf to a histogram
        # thus multiply the pdf value with the bin_width thus getting area of each bar
        # since the pdf value by itself does not make sense unless you actually integrate
        self.E_fn = np.dot(fn_vals, pdf_vals)*bin_width
        return self.E_fn

    def compute_E_fn_using_sampling(self, interactive=False):
        # truncate will always be true

        dpltr = DistributionPlotter()
        fig_no = 1
        target_line = dpltr.plot_scaled_pdf(self.target, fig_no, self.target_distr_str + ' fn scaled pdf', color='b')
        proposal_line = dpltr.plot_scaled_pdf(self.proposal, fig_no, self.proposal_distr_str + ' fn scaled pdf', color='r')
        all_legend = [target_line, proposal_line]
        plt.figure(fig_no)
        plt.legend(all_legend)

        pline = self.plot_function(fig_no)
        all_legend.append(pline)
        figure(fig_no)
        legend(loc='upper left')
        plt.legend(all_legend)

        num_samples = 0
        if interactive:
            num_samples = self.num_samples_int
        else:
            num_samples = self.num_samples

        s_proposal = dpltr.generate_samples(self.proposal, num_samples)

        imp_weights = self.target.pdf(s_proposal)/self.proposal.pdf(s_proposal)

        fn_vals = self.function(s_proposal)
        
        if interactive:
            ion()
            cnt = 0
            first_pass = True
            for sm in s_proposal:
                if sm >= PLOT_LBOUND and sm <= PLOT_RBOUND:
                    line1, = plt.plot(sm, 0, c='g', marker='*', markersize=10, label='sample from proposal')
                    sample_str = 'sample ' + str(sm)
                    plt.title(sample_str)
                    if first_pass:
                        all_legend.append(line1)
                        plt.legend(all_legend)
                        first_pass = False
                    cnt = cnt + 1
                    temp_E_fn_sampling = np.dot(imp_weights[:cnt], fn_vals[:cnt])/cnt
                    # print this on the plot not terminal, how to update plot text ???
                    print temp_E_fn_sampling                
                    draw()
                    pause(0.001)
            ioff()

        self.E_fn_sampling = np.dot(imp_weights, fn_vals)/self.num_samples
        return self.E_fn_sampling

# setting symmetric=True reduces to Metropolis algorithm
# so you have to make sure the proposal distribution is symmetric
class MetropolisHastings(Sampling):
    def __init__(self,
                 target, target_shape, target_loc, target_scale, target_fn_scale,
                 proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale,
                 symmetric=False):
        Sampling.__init__(self,
                          target, target_shape, target_loc, target_scale, target_fn_scale,
                          proposal, proposal_shape, proposal_loc, proposal_scale, proposal_fn_scale)    

        self.symmetric = symmetric

        # number of initial samples to throw away
        self.num_throw = 100


    def generate_target_samples(self, interactive=False):

        dpltr = DistributionPlotter()
        fig_no = 1
        if interactive:
            target_line = dpltr.plot_scaled_pdf(self.target, fig_no, self.target_distr_str + ' fn scaled pdf', color='b')
            all_legend = [target_line]
            plt.figure(fig_no)
            plt.legend(all_legend)
        
        num_samples = 0
        if interactive:
            num_samples = self.num_samples_int
        else:
            num_samples = self.num_samples

        # generating 0 mean cauchy samples, we will add the conditioned point to it
        # to simulate q(x, y) => going from or conditioned on x, generate sample y
        self.s_proposal = dpltr.generate_samples(self.proposal, num_samples)

        unif_d = Distribution.factory("uniform", 0, 0, 1.0, 1.0)
        s_unif = dpltr.generate_samples(unif_d, len(self.s_proposal))

        # start point
        x = unif_d.rvs(1)*(COMP_RBOUND - COMP_LBOUND) - COMP_LBOUND
        x = x[0]

        if interactive:
            ion()
            plt.figure(fig_no)
            print 'Total number of samples ', len(self.s_proposal)

        s_keep = np.zeros(len(self.s_proposal)+1)
        s_keep[0] = x
        i = 0
        first_pass_sample = True
        first_pass_accept = True

        for sm in self.s_proposal:
            y = sm + x

            if interactive:
                line1, = plt.plot(y, 0, c='r', marker='+', markersize=10, label='new point')
                sample_str = 'sample ' + str(y)
                plt.title(sample_str)

                if first_pass_sample:
                    all_legend.append(line1)
                    plt.legend(all_legend)
                    first_pass_sample = False
                draw()
                pause(0.001)

            t_p_scaled_x = self.target.fn_pdf(x)    
            t_p_scaled_y = self.target.fn_pdf(y)

            if not self.symmetric:
                # given x, we assume x is the mean of the gamma
                # x = MEAN + SHAPE*SCALE
                p_p_x_to_y = self.proposal.pdf(y-x)
                p_p_y_to_x = self.proposal.pdf(x-y)
                alpha = (t_p_scaled_y*p_p_y_to_x)/(t_p_scaled_x*p_p_x_to_y)
            else:
                alpha = t_p_scaled_y/t_p_scaled_x

            if (s_unif[i] < alpha):
                x = y
            else:
                x = x

            if interactive:
                line2, = plt.plot(x, 0, c='g', marker='+', markersize=10, label='accepted point')
                accepted_str = 'accepted ' + str(x)
                plt.title(accepted_str)
                if first_pass_accept:
                    all_legend.append(line2)
                    plt.legend(all_legend)
                    first_pass_accept = False
                draw()
                pause(0.001)

            i = i + 1    
            s_keep[i] = x

        s_keep = s_keep[self.num_throw:i+1]    

        fig_no = 2    
        target_line = dpltr.plot_scaled_pdf(self.target, fig_no, self.target_distr_str + ' pdf', scaled=False, color='b')
        pline = dpltr.plot_norm_hist(fig_no, 'sampled norm histogram', s_keep, self.num_bins, color='y')
        all_legend = [target_line, pline]
        plt.figure(fig_no)
        plt.legend(all_legend)
  
        plt.show()
        ioff()


class Gibbs():
    def __init__(self):
        self.data = []

class BlockGibbs(Gibbs):
    def __init__(self):
        self.data = []

class CollapsedGibbs(Gibbs):
    def __init__(self):
        self.data = []


def parse_args(args):

    argp = ArgumentParser(description=__doc__)
    options = argp.parse_args()

    return options


def main(args):
    # Processing the input string
    args = parse_args(args)

    rej = Rejection('gamma', 2.0, 0, 1.0, 1.0, 
                    'cauchy', 0, 0, 4.0, 5.5)
    rej.visualize_basic_plots()
    rej.generate_target_samples(interactive=True)


    # imp = Importance('gamma', 15.0, -20, 2.0, 1.0,
    #                  'cauchy', 0, 0, 5, 1.0)
    # imp.visualize_basic_plots()
    # print 'Brute force E[f] = ', imp.compute_E_fn()
    # print 'Using sampling E[f] = ', imp.compute_E_fn_using_sampling(interactive=False)

    # target distribution scale (different than rejection sampling)
    # this simulates the fact that you don't really need the exact
    # proposal distribution and f(x) \prop p(x) is sufficient
    # the reason is that the acceptance is proportional to the 
    # ratio of the distributions so the normalization cancels

    # m = MetropolisHastings('gamma', 2.0, 0, 1.0, 2.0,
    #                        'cauchy', 0, 0, 4.0, 1.0,
    #                        symmetric=True)
    # m.visualize_basic_plots()
    # m.generate_target_samples(interactive=False)

    # setting self.GAMMA_MEAN = -self.GAMMA_SHAPE*self.GAMMA_SCALE
    # the mean of above gamma distribution is
    # GAMMA_MEAN + GAMMA_SHAPE*GAMMA_SCALE
    # this is done to make the mean of the gamma distribution be 0
    # mh = MetropolisHastings('cauchy', 0, 0, 4, 2.0,
    #                         'gamma', 2.0, -6.0, 3.0, 1.0,
    #                         symmetric=False)
    # mh.visualize_basic_plots()
    # mh.generate_target_samples(interactive=True)
    # Notes:
    # For an asymmetric distribution, had to make sure i shift the distribution so mean is at 0
    # so that there is a non-zero probability of going to the left, otherwise
    # it would always sample points to the right and never to left.


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        sys.stderr.write('Unexpected %s:\n%s\n' % (type(e).__name__, traceback.format_exc()))
    sys.exit(1)

