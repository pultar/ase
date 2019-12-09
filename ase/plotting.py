def get_fig(*args, interactive=False, **kwargs):
    if interactive:
        import matplotlib.pyplot as plt
        return plt.figure(*args, **kwargs)
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(*args, **kwargs)
    canvas = FigureCanvasAgg(fig)
    return fig
