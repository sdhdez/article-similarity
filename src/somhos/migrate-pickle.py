import sys,os
from memory_profiler import memory_usage
import pickle
# Whoosh 
from whoosh.index import create_in
from whoosh.fields import Schema, ID, STORED

def get_name(ext = ".bin"):
    return sys.argv[1][:-len(ext)]

def migrate_large_whoosh():
    fname = get_name()
    print("Pickle path: %s" % fname)
    with open(fname + ".bin", "rb") as fp:
        dic = pickle.load(fp)
        fp.close()
        len_dic = len(dic)
        print("Length: %d" % len_dic)
        if not os.path.exists(fname):
            os.mkdir(fname)
            # Define schema for indexed data 
            schema = Schema(pair=ID(unique=True), # pair 
                            sim=STORED()) # Similarity
            # Create index 
            ix = create_in(fname, schema)
            # Define writer 
            writer = ix.writer(limitmb=2048)
            i = 1
            for k,v in dic.items():
                writer.add_document(pair=k, sim=v) 
                if i % 1000000 == 0:
                    print(i, "pairs")
                i+=1
            print(i, "pairs")
            writer.commit()

def save_part_pickle(fname, p, tmp):
    with open(fname + "/measure-%s.pkl" % p, "wb") as fout:
        pickle.dump(tmp, fout)
        fout.close()

def migrate_large():
    fname = get_name()
    print("Pickle path: %s" % fname)
    with open(fname + ".bin", "rb") as fp:
        dic = pickle.load(fp)
        fp.close()
        len_dic = len(dic)
        print("Length: %d" % len_dic)
        if not os.path.exists(fname):
            os.mkdir(fname)
            tmp = {}
            i = 1
            p = 0
            for k,v in dic.items():
                if v < 1.0:
                    tmp[k] = v
                if i % 500000 == 0:
                    print(i, "pairs")
                    save_part_pickle(fname, p, tmp)
                    p+=1
                    del tmp
                    tmp = {}
                i+=1
            save_part_pickle(fname, p, tmp)

def test_load():
    fname = get_name()
    print("Pickle path: %s" % fname)
    if os.path.exists(fname):
        p = 0
        dic = {}
        el_sum = 0
        while True:
            fpath = fname + "/measure-%s.pkl" % p
            if os.path.exists(fpath):
                with open(fpath, "rb") as fin:
                    tmp = pickle.load(fin)
                    el_sum += len(tmp)
                    dic.update(tmp)
                    del tmp
                p+=1
            else:
                break
        print("Elements:", el_sum)
        print("Elements:", len(dic))

# migrate_large()
mem_usage = memory_usage(test_load)
print("Maximum memory usage: %s " % max(mem_usage))
