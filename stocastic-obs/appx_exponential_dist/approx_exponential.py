import numpy as np
import matplotlib.pyplot as plt
mean_time = 5
lambda_val = 1/mean_time
tau = .02
p = lambda_val * tau
counter_val_list = []
total_iterations = 10000
for iteraion_no in range(total_iterations):
    event_occured = False
    counter_val = 0
    while not event_occured:
        counter_val += 1
        event_occured = True if np.random.binomial(1, p, size=None) == 1 else False
    counter_val_list.append(counter_val)            
exponential_list = np.random.exponential(scale=(1/lambda_val)/tau, size=total_iterations)
plt.hist(counter_val_list)
plt.xlabel('time (in units of .2 seconds)')
plt.ylabel('Count')
plt.title('Approximate distribution')
plt.figure()
plt.hist(exponential_list)
plt.xlabel('time (in units of .2 seconds)')
plt.ylabel('Count')
plt.title('True distribution')
plt.show()