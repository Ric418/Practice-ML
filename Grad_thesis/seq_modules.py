import os

def seq_save(seq, filename, flag):
    if not os.path.exists(filename):
        with open(filename, 'wt') as save:
                    save.write(seq + '\n')
                    #print("New sequence has saved.",end='\n\n')
    else:
        with open(filename, 'rt') as load:
            counter = 1
            while True:
                line = load.readline()
                if line == seq + '\n':
                    #print("There is same sequence.",end='\n\n')
                    break
                elif not line:
                    with open(filename, 'at') as save:
                        save.write(seq + '\n')
                        #print("New sequence has saved.")
                    break
    
    if flag == True:
        #print("final output is ...")
        counter = 1
        with open(filename, 'rt') as load:
            while True:
                line = load.readline()
                if not line:
                    break
                #print(counter,":",line,end='')
                counter+=1
            

def vec2seq_save(filename, seq, batch):
    seq = seq.numpy()
    for itr, data in enumerate(seq):
        sequences = '' #init sequences
        #print("This is", itr+1, "th loop.")
        channel1 = data[0]
        channel2 = data[1]
        channel3 = data[2]
        channel4 = data[3]
        for c1, c2, c3, c4 in zip(channel1, channel2, channel3, channel4):
            if max(c1, c2, c3, c4) == c1:
                sequences += 'A'
            elif max(c1, c2, c3, c4) == c2:
                sequences += 'C'
            elif max(c1, c2, c3, c4) == c3:
                sequences += 'G'
            else:
                sequences += 'T'
        if itr + 1 == batch:
            seq_save(sequences,filename,True)
        else:
            seq_save(sequences,filename,False)
            
def txt2fasta(input_file, output_file, fasta_name):
    '''
    input_text:Fasta形式に変換したい読み込むTXTファイルの名前
    filename:出力するFasta形式のTXTファイルの名前
    fasta_name:Fasta形式の > の次に来る名前
    '''
    with open(input_file, 'rt') as load:
        counter = 1
        while True:
            line = load.readline()
            if not line:
                    break
            else:
                if not os.path.exists(output_file):
                    with open(output_file, 'wt') as save:
                        save.write('>'+ fasta_name + str(counter) + '\n' + line)
                        counter += 1
                else:
                    with open(output_file, 'at') as save:
                        save.write('>'+ fasta_name + str(counter) + '\n' + line)
                        counter += 1