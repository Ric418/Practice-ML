def sequence_save(seq, filename):
    with open(filename, 'rt') as load:
        counter = 1
        while True:
            line = load.readline()
            if line == seq + '\n':
                print("There is same sequence.",end='\n\n')
                break
            elif not line:
                with open(filename, 'at') as save:
                    save.write(seq + '\n')
                    print("New sequence has saved.",end='\n\n')
                break
            else:
                print(counter,":",line,end='')
                counter+=1
                
    print("final output is ...")
    counter = 1
    with open(filename, 'rt') as load:
        while True:
            line = load.readline()
            if not line:
                break
            print(counter,":",line,end='')
            counter+=1
            