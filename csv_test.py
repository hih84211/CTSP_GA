import csv

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
    fields = ['org', '2015', '2014', '2013']
    dw = {'orgname1': {'2015': [2, 3], '2014': 1, '2013': 1},
          'orgname2': {'2015': 1, '2014': 2, '2013': 3},
          'orgname3': {'2015': 1, '2014': 3, '2013': 1}}

    #print(dw['orgname1']['2015'])

    with open("store_data.csv", mode="a", newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for k in dw:
            print(k)
            w.writerow({field: dw[k].get(field) or k for field in fields})
