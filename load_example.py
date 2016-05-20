import cPickle as pickle

from stanza.research.instance import Instance

if __name__ == '__main__':
    with open('runs/speaker_fourier_3d0L/quickpickle.p', 'rb') as infile:
        model = pickle.load(infile)
    print(model.score([Instance((120., 100., 100.), 'green')]))
