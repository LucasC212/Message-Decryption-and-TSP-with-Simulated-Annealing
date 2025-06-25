#import matplotlib.pyplot as plt
from metropolis_hastings import *
import shutil
from deciphering_utils import *

#!/usr/bin/python

import sys
from optparse import OptionParser


def main(argv):
   inputfile = None
   decodefile = None
   parser = OptionParser()

   parser.add_option("-i", "--input", dest="inputfile", 
                     help="input file to train the code on")
   
   parser.add_option("-d", "--decode", dest="decode", 
                     help="file that needs to be decoded")
   
   # changed default iters to 15k
   parser.add_option("-e", "--iters", dest="iterations", type="int", 
                     help="total accepted moves to run", default=15000)
   ################### new code ###################
   # this is inverse temperature, we can just choose a random default 
   # value for this after playing around with what seems to work
   parser.add_option("-b", "--beta0", dest="beta0", type="float",
                     help="intial inverse temp", default=0.3)
   
   # this is the cooling factor
   parser.add_option("-a", "--alpha", dest="alpha", type="float",
                     help="cooling factor alpha (<1)", default=0.995)
   
   # this is how many moves are accepted before we cool the temperature 
   parser.add_option("-k", "--block", dest="block", type="int",
                     help="accepted moves per temp level", default=3000)
   ################### end ###################

   parser.add_option("-t", "--tolerance", dest="tolerance", 
                     help="percentate acceptance tolerance, before we should stop", default=0.02)
   
   parser.add_option("-p", "--print_every", dest="print_every", 
                     help="number of steps after which diagnostics should be printed", default=10000)

   (options, args) = parser.parse_args(argv)

   filename = options.inputfile
   decode = options.decode
   
   if filename is None:
      print("Input file is not specified. Type -h for help.")
      sys.exit(2)

   if decode is None:
      print("Decoding file is not specified. Type -h for help.")
      sys.exit(2)

   char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)
   
   s = list(open(decode, 'r').read())
   scrambled_text = list(s)
   # we are replacing this loop below with our SA loop
   # i = 0
   # initial_state = get_state(scrambled_text, tr, fr, char_to_ix)
   # states = []
   # entropies = []
   # while i < 3:
   #    iters = options.iterations
   #    print_every = int(options.print_every)
   #    tolerance = options.tolerance
   #    state, lps, _ = metropolis_hastings(initial_state, proposal_function=propose_a_move, log_density=compute_probability_of_state, 
   #                                          iters=iters, print_every=print_every, tolerance=tolerance, pretty_state=pretty_state)
   #    states.extend(state)
   #    entropies.extend(lps)
   #    i += 1
      #if(i<3): input("\n Starting in a new Random State...")
   
   ################### new code here as well ###################
   # define the variables based on the inputs
   beta = float(options.beta0)      # inverse temperature
   alpha = float(options.alpha)     # cooling factor, a scalar (0, 1)
   budget = int(options.iterations) # total number of MCMC steps
   block = int(options.block)       # number of steps between each temperature update

   # start at our initial state with no entropies
   initial_state = get_state(scrambled_text, tr, fr, char_to_ix)
   state = initial_state
   states = [state]
   entropies = []

   
   while budget > 0:
      # run up to 'block' MCMC steps or budget if it's smaller
      iters = min(block, budget)

      # log probability of being in a state, multiplied by scheduled beta
      logdens = lambda st: beta * compute_probability_of_state(st)

      # run metropolis hastings algorithm with our logdens (which includes the beta)
      state_run, lps_run, _ = metropolis_hastings(
         state, proposal_function=propose_a_move, log_density=logdens,
         iters=iters, print_every=int(options.print_every), tolerance=options.tolerance,
         pretty_state=pretty_state
      )

      state = state_run[-1]
      states.extend(state_run[1:])
      entropies.extend(lps_run)
      budget -= iters

      # cool the system (decrease temperature --> increase beta)
      beta /= alpha
      print(f"[SA] Beta updated to {beta:.4f}")

   ################### end code ###################
   
   p = list(zip(states, entropies))
   p.sort(key=lambda x:x[1])
   
   print(" Best Guesses : \n")
   
   for j in range(1,4):
      print(f"Guess {j}: \n")
      print(pretty_state(p[-j][0], full=True))
      print(shutil.get_terminal_size().columns*'*')
   
if __name__ == "__main__":
   main(sys.argv)