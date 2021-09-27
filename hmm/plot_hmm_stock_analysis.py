"""
Gaussian HMM of stock data
--------------------------

This script shows how to use Gaussian HMM on stock price data from
Yahoo! finance. For more information on how to visualize stock prices
with matplotlib, please refer to ``date_demo1.py`` of matplotlib.
"""

from __future__ import print_function

import sys
import io
import os

import datetime

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.dates import DayLocator

# try:
#     from matplotlib.finance import quotes_historical_yahoo_ochl

# except ImportError:
#     # For Matplotlib prior to 1.5.
#     from matplotlib.finance import (
#         quotes_historical_yahoo as quotes_historical_yahoo_ochl
#     )
# from mpl_finance import quotes_historical_yahoo_ochl
from hmmlearn.hmm import GaussianHMM

sys.stdout = io.TextIOWrapper(
    sys.stdout.buffer, encoding='utf8')  # Change default encoding to utf8

print(__doc__)


###############################################################################
# Get quotes from Yahoo! finance
def load_sample_stock():
    import csv
    ret = []
    with open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'sample_stock.csv'),
            newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            ret.append((datetime.datetime.strptime(row["Date"], "%d/%m/%Y"),
                        float(row["Close"]), float(row["Volume"])))
            # ret.append((np.datetime64(row["Date"]), float(row["Close"]), float(row["Volume"])))
    return ret


quotes = load_sample_stock()
# quotes = quotes_historical_yahoo_ochl(
#    "INTC", datetime.date(1995, 1, 1), datetime.date(2012, 1, 6))

# Unpack quotes
dates = np.array([q[0] for q in quotes], dtype=datetime.datetime)
close_v = np.array([q[1] for q in quotes])
volume = np.array([q[2] for q in quotes])[1:]

# Take diff of close value. Note that this makes
# ``len(diff) = len(close_t) - 1``, therefore, other quantities also
# need to be shifted by 1.
diff = np.diff(close_v)
dates = dates[1:]
close_v = close_v[1:]

# Pack diff and volume for training.
X = np.column_stack([diff, volume])

###############################################################################
# Run Gaussian HMM
print("fitting to HMM and decoding ...", end="")

# Make an HMM instance and execute fit
model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=1000)
model.fit(X)

###############################################################################
# Print trained parameters and plot
print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
means_ = []
vars_ = []
for i in range(model.n_components):
    means_.append(model.means_[i][0])
    vars_.append(np.diag(model.covars_[i])[0])
    print("第{0}号隐藏状态".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

# Predict the optimal sequence of internal hidden state
X = X[-26:]
dates = dates[-26:]
close_v = close_v[-26:]

hidden_states = model.predict(X)
predict_means = []
predict_vars = []
for idx, hid in enumerate(range(model.n_components)):
    comp = np.argmax(model.transmat_[idx])
    predict_means.append(means_[comp])
    predict_vars.append(vars_[comp])

# print(X)
print(hidden_states)
print(model.means_)
print(model.covars_)
print("done")

fig, axs = plt.subplots(model.n_components + 1, sharex=True, sharey=True)
for i, ax in enumerate(axs[:-1]):
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.set_title("{0}th hidden state".format(i))

    # Use fancy indexing to plot data in each state.
    mask = hidden_states == i
    yesterday_mask = np.concatenate(([False], mask[:-1]))
    if len(dates[mask]) <= 0:
        continue
    if predict_means[i] > 0.01:
        ax.plot_date(dates[mask], close_v[mask], "^", c="#FF0000")
    elif predict_means[i] < -0.01:
        ax.plot_date(dates[mask], close_v[mask], "v", c="#00FF00")
    else:
        ax.plot_date(dates[mask], close_v[mask], "+", c="#000000")

    # Format the ticks.
    ax.xaxis.set_minor_locator(DayLocator())

    ax.grid(True)
    ax.legend(
        ["Mean: %0.3f\nvar:   %0.3f" % (predict_means[i], predict_vars[i])],
        loc='center left',
        bbox_to_anchor=(1, 0.5))

axs[-1].plot_date(dates, close_v, "-", c='#000000')
axs[-1].grid(True)
# box = axs[-1].get_position()
# axs[-1].set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.xaxis.set_minor_locator(DayLocator())

fig.autofmt_xdate()
plt.subplots_adjust(
    left=None, bottom=None, right=0.75, top=None, wspace=None, hspace=0.43)

plt.show()
