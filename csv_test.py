import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class CsvWriter:
    def __init__(self, file):
        self.headers = ['populationSize',
                        'gen', 'bestFit', 'avgFit', 'genRuntime']
        self.file = file
        with open(self.file, mode="a", newline='') as f:
            w = csv.DictWriter(f, self.headers)
            w.writeheader()

    def write(self, data):
        with open(self.file, mode="a", newline='') as f:
            w = csv.DictWriter(f, self.headers)
            w.writerow({field: data.get(field) for field in self.headers})


if __name__=='__main__':
    SMALL_SIZE = 8
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 20

    var = pd.read_excel('for_plot.xlsx')
    # print(var)

    x = list(var['genRuntime'])
    y = list(var['bestFit'])

    plt.boxplot(y)
    plt.show()

    #plt.figure(figsize=(15, 15))
    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    #plt.style.use('seaborn')
    #plt.style.use('_mpl-gallery')
    # plt.scatter(x, y, marker="*", s=100, edgecolors="black", c="yellow")
    # plt.scatter(x, y, edgecolors="black",)
    fig, ax2 = plt.subplots()
    ax2.scatter(x, y, edgecolors="black")
    plt.show()
    # plt.xlabel("Runtime in sec")
    # plt.ylabel("Distance in pixel unit")
    # plt.title("Some title here")

