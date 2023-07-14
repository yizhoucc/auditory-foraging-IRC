# Modeling sense of time

## Data collection:
- Ask the user for a target time.
- Ask the user to press space bar to start the hidden timer. 
- User presses space bar again when the user feels just target time has passed by.
- Make note of the actual time that had passed by.
- Repeat the above to collect data points until user loses interest.
- Store the data in the 'store' folder in the form of dictionary.
- The dictionay keys are the different target times chosen by the user.
- The corresponding value is a list of the corresponding actual times.

## Data analysis:
- For a given target time, we assume that the collected data follows a gamma distribution with mean as the target time.
- Compute the number of nodes in our graphical model that would best fit the empirical distribution (using MLE maybe).
- Find out the dependence between chosen time and best fitting number of nodes.