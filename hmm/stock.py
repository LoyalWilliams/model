import datetime
from matplotlib import dates
import numpy as np
from matplotlib import cm, pyplot as plt
from matplotlib.dates import DayLocator
from hmmlearn.hmm import GaussianHMM


def load_sample_stock():
    import csv
    ret = []
    with open('hmm/sample_stock.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            ret.append((datetime.datetime.strptime(
                row['Date'], "%Y/%m/%d"), float(row['Close']), float(row['Volume'])))
    return ret


quotes = load_sample_stock()
dates = np.array([q[0] for q in quotes], dtype=datetime.datetime)[1:]
close_v = np.array([q[1] for q in quotes])
volume = np.array([q[2] for q in quotes])[1:]

diff = np.diff(close_v)

X = np.column_stack([diff, volume])

model = GaussianHMM(n_components=10, covariance_type='diag', n_iter=1000)
model.fit(X)

print('model.means_:')
print(model.means_)

print('model.covars_:')
print(model.covars_)

X = X[-26:]
dates = dates[-26:]
close_v = close_v[-26:]
hidden_states = model.predict(X)

means_ = []
vars_ = []
for i in range(model.n_components):
    means_.append(model.means_[i][0])
vars_.append(np.diag(model.covars_[i])[0])

predict_means = []
for idx, hid in enumerate(range(model.n_components)):
    comp = np.argmax(model.transmat_[idx])
    predict_means.append(means_[comp])

fig, axs = plt.subplots(model.n_components+1, sharex=True, sharey=True)
for i, ax in enumerate(axs[:-1]):
    ax.set_title('{0}th hidden state'.format(i))

    mask = hidden_states == i
    yesterday_mask = np.concatenate(([False], mask[:-1]))
    # print(mask)
    # print(dates)
    if len(dates[mask]) <= 0:
        continue
    if predict_means[i] > 0.01:
        ax.plot_date(dates[mask], close_v[mask], "^", c="#FF0000")
    elif predict_means[i] < -0.01:
        ax.plot_date(dates[mask], close_v[mask], 'v', c='#00FF00')
    else:
        ax.plot_date(dates[mask], close_v[mask], '+', c='#000000')

    ax.xaxis.set_minor_locator(DayLocator())

    ax.grid(True)
    # ax.legend([])

axs[-1].plot_date(dates, close_v, '-', c='#000000')
axs[-1].grid(True)

fig.autofmt_xdate()
plt.subplots_adjust(left=None, bottom=None, right=0.75,
                    top=None, wspace=None, hspace=0.43)

print(model.transmat_)
print(predict_means)
print(hidden_states)
print(len(hidden_states))
plt.show()