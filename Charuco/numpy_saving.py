
import numpy as np

def main():
    data2 = np.load('teemp.npy', allow_pickle=True)
    dict = data2[()]
    print('donje')

if __name__ == '__main__':
    main()