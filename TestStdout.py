import sys
import time

if __name__ == '__main__':
    i = 0
    while i < 100:
        i += 1
        print('print' + str(i))
        time.sleep(1)
        sys.stdout.write('stdout' + str(i) + '\r')
        sys.stdout.flush()
