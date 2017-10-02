import quandl
from sklearn.preprocessing import MinMaxScaler
import neurokit as nk
from pynamical import simulate, save_fig, phase_diagram, phase_diagram_3d
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import matplotlib.cm as cm, matplotlib.font_manager as fm



quandl.ApiConfig.api_key = 'h6tfVg1ps54v4bcpc3xz'
START = 2500
END = 3000

TICKERS = ['AAPL', 'AMZN', 'CMCSA', 'F', 'MAR', 'CVX']
LAGS = [1, 5, 10] # time lag to plot the phase diagram


def get_title_font(family='Helvetica', style='normal', size=20, weight='normal', stretch='normal'):
    """
    Define fonts to use for image titles.

    Arguments
    ---------
    family : string
    style : string
    size : numeric
    weight : string
    stretch : string

    Returns
    -------
    matplotlib.font_manager.FontProperties
    """

    if family == 'Helvetica':
        family = ['Helvetica', 'Arial', 'sans-serif']
    fp = fm.FontProperties(family=family, style=style, size=size, weight=weight, stretch=stretch)
    return fp


def get_label_font(family='Helvetica', style='normal', size=12, weight='normal', stretch='normal'):
    """
    Define fonts to use for image axis labels.

    Arguments
    ---------
    family : string
    style : string
    size : numeric
    weight : string
    stretch : string

    Returns
    -------
    matplotlib.font_manager.FontProperties
    """

    if family == 'Helvetica':
        family = ['Helvetica', 'Arial', 'sans-serif']
    fp = fm.FontProperties(family=family, style=style, size=size, weight=weight, stretch=stretch)
    return fp

def plot_phase_plot(pops, lags = LAGS, xmin=0, xmax=1, ymin=0, ymax=1,title='',
                    xlabel='Population (t)', ylabel='Population (t + 1)',
                    marker='.', size=5, alpha=0.7, color='#003399',
                    filename='image', folder='images', dpi=300, bbox_inches='tight',
                    pad=0.1, figsize=(6,6), save=True, show=False):

    # create a new matplotlib figure and axis and set its size
    fig, (ax1,ax2, ax3) = plt.subplots(figsize=figsize, nrows=3, ncols=1)


    index = 0
    for ax in (ax1,ax2, ax3):
        # set the plot title, x- and y-axis limits, and x- and y-axis labels
        ax.set_title(title, fontproperties=get_title_font())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel, fontproperties=get_label_font())
        ax.set_ylabel("Population (t+{0})".format(LAGS[index]), fontproperties=get_label_font())
        # TODO: hardcoded values for scatter plot
        ax.scatter(pops['open_{0}'.format(LAGS[index])], pops['close_{0}'.format(LAGS[index])], marker=marker,
                   edgecolor='none', s=size, alpha=alpha)
        index += 1

    if save:
        save_fig(filename=filename, folder=folder, dpi=dpi, bbox_inches=bbox_inches, pad=pad)
    if show:
        plt.show()
    return fig, ax



LYAP = []

for i in range(len(TICKERS)):
    data = quandl.get_table('WIKI/PRICES', ticker=TICKERS[i])

    # data scaling
    scaler = MinMaxScaler()


    pops = []
    for idx, lag in enumerate(LAGS):
        open_array = data[START:END]['open']
        close_array = data[(START+lag):(END+lag)]['close']
        # open_array = np.expand_dims(open_array, axis=1)
        # close_array = np.expand_dims(close_array, axis=1)
        # tmp_pops = np.concatenate((open_array, close_array), axis=1)
        # tmp_pops = scaler.fit_transform(tmp_pops)
        open_array = scaler.fit_transform(open_array)
        pops.append(open_array)
        close_array = scaler.fit_transform(close_array)
        pops.append(close_array)

    pops = pd.DataFrame(np.transpose(np.asarray(pops)), columns=["open_{0}".format(LAGS[0]), "close_{0}".format(LAGS[0]),
                                       "open_{0}".format(LAGS[1]), "close_{0}".format(LAGS[1]),
                                       "open_{0}".format(LAGS[2]), "close_{0}".format(LAGS[2])])


    # phase_diagram(pops, size=20, color=['#003399', '#cc0000'], show = False, ymax=1.005, legend=True,
    #               filename='phase_diagram_{0}_lag={1}'.format(TICKERS[i],LAG))

    plot_phase_plot(pops=pops,filename='phase_diagram_{0}_lags={1}'.format(TICKERS[i],str(LAGS)), show=False)

    LYAP.append(nk.complexity(
        pops.close_1.as_matrix(),lyap_r=True, lyap_e=False, sampling_rate = 1000, shannon = False,
        sampen = False, multiscale = False, spectral = False, svd = False, correlation = True,
        higushi = False, petrosian = False, fisher = False, hurst = False, dfa = False,
        emb_dim = 2, tolerance = "default", k_max = 8, bands = None, tau = 1))

print(TICKERS)
print(LYAP)