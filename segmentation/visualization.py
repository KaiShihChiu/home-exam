import matplotlib.pyplot as plt

def plot_histogram(data, title, xlabel, ylabel, color='blue', bins=20):
    """
    Plot a histogram of the given data.
    
    Args:
        data (list or np.array): Data to plot.
        title (str): Title of the histogram.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        color (str): Color of the histogram bars.
        bins (int): Number of bins in the histogram.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, color=color, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.show()

def display_image(image, title, cmap='gray'):
    """
    Display an image using Matplotlib.
    
    Args:
        image (np.array): Image to display.
        title (str): Title of the image.
        cmap (str): Colormap to apply (default is 'gray').
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    # plt.show()
