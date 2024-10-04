# Mu-MIMO-Uplink
The file MU_MIMO.py is the uplink system model. To run the simulation for uplink transmisison, run the following:

`python run.py`

The simulation includes:
- Multiple time slot iterations
- User movement via random walk per time slot
- Uniform power allocation
- Subcarrier allocation via round robin (Random allocation is also included)
- Outputs average data rates, SNRs, and MSEs for all users after execution.

You may edit the values for the simulation, like the number of time slots, the BW, maximum users' power etc. inside run.py file.