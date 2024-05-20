import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Setup for plotting in Latex. This will affect all plots after this has been imported.
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": 8,
    "font.size": 10,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8
}

plt.rcParams.update(tex_fonts)

def calculate_size(width=411.4, fraction=1):
    # Taken from: https://jwalton.info/Embed-Publication-Matplotlib-Latex/
    # Use with latex: \showthe\textwidth
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim

def save_plot(fig, filename, caption=""):
    """
    Saves a plot to disk, along with an optional caption in a sidecar
    tex file.
    """
    fig.savefig(filename, format='pdf', bbox_inches='tight', dpi=300)

    # Write the caption to a file
    if caption:
        caption_filename = filename.replace(".pdf", ".tex")
        with open(caption_filename, 'w') as f:
            f.write(caption)

def plot_trace(solution, title="", ax=None, second_axis=True, figure_size=(8, 6)):
    """
    Plots the trace of a solution.
    ax: An existing axis to plot on. If not provided, a new figure and
        axis will be provided.
    title: Title to be placed on the plot.
    second_axis: Should extra data be plotted on a second y-aix? Can make the
        graph busy.
    figure_size: tuple of (x_dim, y_dim). Only used if ax not provided.
    """

    # From the solution's associated problem,
    # determine if there is a greedy solution
    # and a known optimal solution.
    problem = solution.problem
    greedy_aspl = None
    optimal_aspl = None 

    title = f"{solution.method}, $|S|$ = {len(problem.S)}, k = {problem.k}" if not title else title

    for s in problem.solutions:
        if s.method == 'Greedy Solver':
            greedy_aspl = s.aspl
        if s.is_optimal:
            optimal_aspl = s.aspl

    df = pd.DataFrame.from_records(solution.trace)
    df = df.rename(columns={'aspl': 'ASPL', 'temperature': 'Temperature', 'solutions_explored': 'Solutions Explored'})
    df['Iteration'] = df.index

    # If ax is provided, plot onto that axis
    if not ax:
        fig, ax1 = plt.subplots(figsize=figure_size)
    else:
        ax1 = ax
        
    # Alternative to a scatter plot
    sns.histplot(df, x="Iteration", y="ASPL", ax=ax1, bins=(100,100), legend=False)
    
    if second_axis:
        ax2 = ax1.twinx()
        if  'temperature' in solution.trace[0]:
            sns.lineplot(df, x="Iteration", y="Temperature", label="Temperature", color='orange', n_boot=0, estimator=None, ax=ax2)
        elif 'solutions_explored' in solution.trace[0]:
            sns.lineplot(df, x="Iteration", y="Solutions Explored", color='orange', label="Solutions Explored", n_boot=0, estimator=None, ax=ax2)
        ax2.legend(loc="upper right", framealpha=0.8)
    ax1.set_title(title, fontsize=8)   
    #plt.title(title, fontsize=8)

    # Indicate the lowest point in the graph
    min_aspl = df['ASPL'].min()
    min_aspl_row = df[df['ASPL'] == min_aspl]
    min_aspl_iteration = min_aspl_row['Iteration'].values[0]
    ax1.axvline(min_aspl_iteration, color='darkblue', linestyle='dashed', alpha=0.9, label="Solver Minimum ASPL")
    ax1.axhline(solution.aspl, color='darkblue', linestyle='dashed', alpha=0.9)

    # Add horizontal lines for greedy and optimal solutions
    if greedy_aspl is not None:
        ax1.axhline(greedy_aspl, color='red', linestyle='solid', alpha=0.8, label="Greedy Solution")
    if optimal_aspl is not None:
        ax1.axhline(optimal_aspl, color='green', linestyle='dotted', label="Optimal Solution")
    ax1.legend(loc="upper left", framealpha=1)
    
    if not ax:
        return fig


def plot_aspls(problem, title="", ax=None):
    # Given a problem, plot the ASPL of each solution
    # in the problem.

    # If ax is provided, plot onto that axis
    if not ax:
        fig, ax1 = plt.subplots(figsize=(8, 6))
    else:
        ax1 = ax

    ax1.set_ylabel("Method")
    ax1.set_xlabel("ASPL")
    data = {'Method': [s.method for s in problem.solutions], 'ASPL': [s.aspl for s in problem.solutions]}
    df = pd.DataFrame(data)
    df = df.sort_values(by='ASPL', ascending=True)
    df = df.reset_index(drop=True)

    y_min = max(df['ASPL'].min() * 0.95, 0)
    y_max = df['ASPL'].max() * 1.05

    sns.barplot(data=df, y='Method', x='ASPL', hue='Method', ax=ax1, order=df['Method'])
    # Set the y axis reasonable limits
    ax1.set_xlim(y_min, y_max)
    #ax1.legend(loc="upper left")
    plt.title(title)

    return fig

def plot_solutions(problem, title="", ncols=2, second_axis=False, figure_size=None):
    # Given a set of solutions to a problem, run plot_trace
    # on each and place the results in a grid.
    # second_axis determines if additional information specific
    # to each problem should be plotted.
    # If figure_size is provided, it should be a tuple of (x_dim, y_dim),
    # else the size is automaticall determined

    # For single solution traces, use plot_trace

    # Determine how many plots we want: ignore solutions that don't
    # have an interesting trace.
    solutions = [s for s in problem.solutions if s.trace and len(s.trace) > 10]
    n = len(solutions)
    assert n > 1, "Function should be used where there are multiple solutions to compare."

    # Determine the number of rows and columns required in our grid
    ncols = min(ncols, n)
    nrows = n // ncols if n % ncols == 0 else n // ncols + 1

    # Calculate optimal plot size based on latex document width
    if not figure_size:
        x_size, y_size = calculate_size()
        #x_size = 7
        y_size = y_size / ncols * nrows * 1.5
        print("Setting x_size to", x_size, "and y_size to", y_size)
    else:
        x_size, y_size = figure_size

    # Configure the subplots that plot_trace will operate on.
    # Share a Y axis across plots in the some row to maximise real estate

    fig, axs = plt.subplots(nrows, ncols, sharey=True, sharex = True, figsize=(x_size, y_size))

    # Disable all the axes for the plots; reenable the ones we explicitly use
    for ax in axs.flatten():
        ax.axis('off')

    for solution, ax in zip(solutions, axs.flatten()):
        ax.axis('on')

        # Get the column it is in. Use this to control whether Y labels should be shown
        plot_trace(solution, title=solution.method, ax=ax, second_axis=second_axis)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    #plt.subplots_adjust(top=0.9) 
    return fig

