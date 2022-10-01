from torch.utils.data import Dataset
import numpy as np

def getSeqs(path):
    """根据.fa文件获取序列"""
    seqs = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if line[0] != '>':
                seqs.append(line)
            line = f.readline()

    return seqs


def one_hot_encode(seqs):
    seqs_array = np.zeros((len(seqs), 72, 4))

    count = 0
    for seq in seqs:
        for i in range(len(seq)):
            if seq[i] == 'A':
                seqs_array[count][i][0] = 1
            elif seq[i] == 'T':
                seqs_array[count][i][1] = 1
            elif seq[i] == 'G':
                seqs_array[count][i][2] = 1
            elif seq[i] == 'C':
                seqs_array[count][i][3] = 1

        count = count + 1

    return seqs_array


def getLabel(path):
    label = []
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            if 'Frankia symbiont of Datisca glomerata chromosome' in line:
                label.append(0)
            elif 'Hydrogenobaculum sp. Y04AAS1 chromosome' in line:
                label.append(1)
            elif 'Candidatus Midichloria mitochondrii IricVA chromosome' in line:
                label.append(2)
            elif 'Corynebacterium variabile DSM 44702 chromosome' in line:
                label.append(3)
            elif 'Psychromonas ingrahamii 37 chromosome' in line:
                label.append(4)
            elif 'Roseiflexus castenholzii DSM 13941 chromosome' in line:
                label.append(5)
            elif 'Alteromonas macleodii str' in line:
                label.append(6)
            elif 'Denitrovibrio acetiphilus DSM 12809 chromosome' in line:
                label.append(7)
            elif 'Sphingomonas wittichii RW1 chromosome' in line:
                label.append(8)
            elif 'Baumannia cicadellinicola str. Hc (Homalodisca coagulata)' in line:
                label.append(9)

            line = f.readline()

    return np.array(label)


def transferLabel(label):
    if label == 0:
        return 'Frankia symbiont of Datisca glomerata chromosome, complete genome'
    elif label == 1:
        return 'Hydrogenobaculum sp. Y04AAS1 chromosome, complete genome'
    elif label == 2:
        return 'Candidatus Midichloria mitochondrii IricVA chromosome, complete genome'
    elif label == 3:
        return 'Corynebacterium variabile DSM 44702 chromosome, complete genome'
    elif label == 4:
        return 'Psychromonas ingrahamii 37 chromosome, complete genome'
    elif label == 5:
        return 'Roseiflexus castenholzii DSM 13941 chromosome, complete genome'
    elif label == 6:
        return 'Alteromonas macleodii str. \'Deep ecotype\' chromosome, complete genome'
    elif label == 7:
        return 'Denitrovibrio acetiphilus DSM 12809 chromosome, complete genome'
    elif label == 8:
        return 'Sphingomonas wittichii RW1 chromosome, complete genome'
    elif label == 9:
        return 'Baumannia cicadellinicola str. Hc (Homalodisca coagulata), complete genome'





class SeqDataset(Dataset):
    def __init__(self, seqs, label):
        self.seqs = seqs
        self.label = label

    def __getitem__(self, item):
        return self.seqs[item], self.label[item]

    def __len__(self):
        return len(self.seqs)