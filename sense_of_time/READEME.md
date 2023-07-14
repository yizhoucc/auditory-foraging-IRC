# Modeling sense of time

## Data collection:
- Ask the user for a target time.
- Ask the user to press space bar to start the hidden timer. 
- User presses space bar again when the user feels just target time has passed by.
- Make note of the actual time that had passed by.
- Repeat the above to collect enough data points until user loses interest.
- Store the data in 'store' folder.

## Data analysis:
- For a given target time, we assume that the collected data follows a gamma distribution with mean as the target time.
- Compute the number of nodes in our graphical model that would best fit the empirical distribution (using MLE maybe).
- Find out the dependence between chosen time and best fitting number of nodes.