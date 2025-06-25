# Message-Decryption-and-TSP-with-Simulated-Annealing
This report improves upon a Metropolis Hastings algorithm by implementing simulated annealing to decrypt an encrypted message. The application is also generalized to the traveling salesman problem (TSP) known in optimization. 

To run the baseline Metropolis Hastings algorithm: python3 run_deciphering.py -i data/warpeace_input.txt -d secret_message.txt

To run the simulated annealing algorithm as was done in the report: python3 run_deciphering_sa.py -i data/warpeace_input.txt -d secret_message.txt -e 50000 -b 0.3 -a 0.995 -k 5000
